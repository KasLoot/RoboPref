from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

from simulation.benchmark.catalog import build_catalog
from simulation.benchmark.generator import _scenario_manifest
from simulation.benchmark.render import (
    _draw_background,
    _extract_objects,
    _is_observation_occluded,
    render_initial,
)


MUJOCO_AVAILABLE = (
    importlib.util.find_spec("mujoco") is not None
    and importlib.util.find_spec("numpy") is not None
)


def _scenario(
    family: str,
    *,
    variant: str,
    seed: int = 37,
):
    return next(
        scenario
        for scenario in build_catalog(families=[family], seeds=[seed])
        if scenario.scene_variant == variant and scenario.outcome == "success"
    )


class SyntheticVisualQualityTests(unittest.TestCase):
    def test_category_mats_use_high_contrast_mustard_and_teal(self) -> None:
        image = _draw_background("category_sort")
        mustard = ImageStat.Stat(image.crop((72, 205, 278, 407))).mean
        teal = ImageStat.Stat(image.crop((362, 205, 572, 407))).mean

        self.assertGreater(
            mustard[0] - mustard[1],
            30,
            "the left sorting mat no longer reads as mustard yellow",
        )
        self.assertGreater(
            mustard[1] - mustard[2],
            60,
            "the mustard mat is too pale to separate light objects",
        )
        self.assertGreater(
            teal[1] - teal[0],
            35,
            "the right sorting mat no longer reads as teal",
        )
        self.assertGreater(
            teal[2] - teal[0],
            35,
            "the teal mat is too neutral to separate light objects",
        )
        self.assertLess(
            abs(teal[1] - teal[2]),
            35,
            "the right mat has drifted away from the teal colour family",
        )

    def test_fallback_worktop_has_warm_nonuniform_wood_grain(self) -> None:
        image = _draw_background("block_stack")
        # This crop is an unobstructed part of the worktop, away from the wall,
        # plank borders, and block-stack contact ellipse.
        worktop = image.crop((120, 160, 510, 290))
        statistics = ImageStat.Stat(worktop)

        self.assertGreater(statistics.mean[0] - statistics.mean[1], 35)
        self.assertGreater(statistics.mean[1] - statistics.mean[2], 25)
        self.assertGreater(
            min(statistics.stddev),
            5,
            "the worktop has regressed to a flat colour",
        )

        adjacent_difference = ImageChops.difference(
            worktop.crop((1, 0, worktop.width, worktop.height)),
            worktop.crop((0, 0, worktop.width - 1, worktop.height)),
        ).convert("L")
        histogram = adjacent_difference.histogram()
        changed_fraction = 1.0 - histogram[0] / sum(histogram)
        self.assertGreater(
            changed_fraction,
            0.02,
            "the procedural horizontal wood grain is no longer visible",
        )

    def test_full_synthetic_render_is_deterministic_without_a_golden_hash(
        self,
    ) -> None:
        scenario = _scenario("category_sort", variant="front_row")

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            render_initial(scenario, first)
            render_initial(scenario, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with Image.open(first) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (640, 480))

    def test_generated_manifest_dict_uses_real_endpoint_and_occlusion_paths(
        self,
    ) -> None:
        success = _scenario("place_setting", variant="front_scatter")
        manifest = _scenario_manifest(success, backend="synthetic")
        initial, initial_supplied = _extract_objects(manifest, "initial")
        final, final_supplied = _extract_objects(manifest, "final")

        self.assertTrue(initial_supplied)
        self.assertTrue(final_supplied)
        self.assertEqual(len(initial), len(success.initial_objects))
        self.assertEqual(len(final), len(success.final_objects))

        unknown = next(
            scenario
            for scenario in build_catalog(
                families=["place_setting"],
                seeds=[37],
            )
            if scenario.scene_variant == "front_scatter"
            and scenario.outcome == "unknown"
        )
        unknown_manifest = _scenario_manifest(unknown, backend="synthetic")
        self.assertTrue(_is_observation_occluded(unknown_manifest))


@unittest.skipUnless(
    MUJOCO_AVAILABLE,
    "optional MuJoCo and NumPy simulation dependencies are absent",
)
class MujocoVisualQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        import numpy as np

        from simulation.benchmark.mujoco_render import (
            HEIGHT,
            WIDTH,
            _build_model,
        )

        cls.mujoco = mujoco
        cls.np = np
        cls.category_scenario = _scenario(
            "category_sort",
            variant="front_row",
        )
        cls.place_scenario = _scenario(
            "place_setting",
            variant="front_scatter",
        )
        cls.block_scenario = _scenario(
            "block_stack",
            variant="wide_scatter",
        )

        cls.category = cls._render_diagnostics(
            _build_model,
            cls.category_scenario,
            WIDTH,
            HEIGHT,
        )
        cls.place = cls._render_diagnostics(
            _build_model,
            cls.place_scenario,
            WIDTH,
            HEIGHT,
        )
        cls.blocks = cls._render_diagnostics(
            _build_model,
            cls.block_scenario,
            WIDTH,
            HEIGHT,
        )

    @classmethod
    def _render_diagnostics(
        cls,
        build_model,
        scenario,
        width: int,
        height: int,
    ):
        mujoco = cls.mujoco
        model = build_model(scenario, scenario.initial_objects)
        data = mujoco.MjData(model)
        home_key = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            renderer.update_scene(data, camera="third_person")
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="third_person")
            segmentation = renderer.render().copy()
        finally:
            renderer.close()
        return model, rgb, segmentation

    def _geom_id(self, model, name: str) -> int:
        geom_id = self.mujoco.mj_name2id(
            model,
            self.mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        self.assertGreaterEqual(geom_id, 0, f"missing geom {name!r}")
        return int(geom_id)

    def _material_id_for_geom(self, model, name: str) -> int:
        material_id = int(model.geom_matid[self._geom_id(model, name)])
        self.assertGreaterEqual(
            material_id,
            0,
            f"geom {name!r} has no named material",
        )
        return material_id

    def _material_name_for_geom(self, model, name: str) -> str:
        material_id = self._material_id_for_geom(model, name)
        return str(
            self.mujoco.mj_id2name(
                model,
                self.mujoco.mjtObj.mjOBJ_MATERIAL,
                material_id,
            )
        )

    def _geom_names(self, model) -> set[str]:
        return {
            str(name)
            for geom_id in range(model.ngeom)
            if (
                name := self.mujoco.mj_id2name(
                    model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
            )
        }

    def _object_mask(self, model, segmentation, object_id: str):
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if (
                self.mujoco.mj_id2name(
                    model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or ""
            ).startswith(f"{object_id}_")
        ]
        self.assertTrue(geom_ids, f"no geoms found for {object_id!r}")
        return self.np.isin(segmentation[:, :, 0], geom_ids)

    def test_camera_is_opposite_the_robot_and_looks_through_the_workspace(self) -> None:
        from simulation.benchmark.mujoco_render import (
            THIRD_PERSON_CAMERA_POSITION,
            THIRD_PERSON_CAMERA_TARGET,
        )

        model = self.blocks[0]
        camera_id = self.mujoco.mj_name2id(
            model,
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            "third_person",
        )
        self.assertGreaterEqual(camera_id, 0)
        position = model.cam_pos[camera_id]
        self.np.testing.assert_allclose(
            position,
            THIRD_PERSON_CAMERA_POSITION,
            atol=1e-9,
        )
        self.assertAlmostEqual(float(position[1]), 0.0)
        self.assertGreater(
            float(position[0]),
            0.495,
            "camera must remain beyond the table's +x edge, opposite the robot",
        )

        expected_forward = (
            self.np.asarray(THIRD_PERSON_CAMERA_TARGET)
            - self.np.asarray(THIRD_PERSON_CAMERA_POSITION)
        )
        expected_forward /= self.np.linalg.norm(expected_forward)
        camera_rotation = model.cam_mat0[camera_id].reshape(3, 3)
        optical_axis = -camera_rotation[:, 2]
        self.assertGreater(
            float(self.np.dot(optical_axis, expected_forward)),
            0.999999,
        )
        self.assertLess(
            float(optical_axis[0]),
            0.0,
            "camera must face back toward the robot at the world origin",
        )

    def test_workspace_uses_textured_nonwhite_materials(self) -> None:
        from simulation.benchmark.mujoco_render import MATERIAL

        category_model = self.category[0]
        place_model = self.place[0]
        self.assertEqual(
            self._material_name_for_geom(
                category_model,
                "benchmark_table_surface",
            ),
            MATERIAL["wood"],
        )
        self.assertEqual(
            self._material_name_for_geom(category_model, "left_region_geom"),
            MATERIAL["yellow_mat"],
        )
        self.assertEqual(
            self._material_name_for_geom(category_model, "right_region_geom"),
            MATERIAL["teal_mat"],
        )
        self.assertEqual(
            self._material_name_for_geom(place_model, "placemat_geom"),
            MATERIAL["yellow_mat"],
        )

        wood_id = self._material_id_for_geom(
            category_model,
            "benchmark_table_surface",
        )
        yellow_id = self._material_id_for_geom(
            category_model,
            "left_region_geom",
        )
        self.assertTrue(
            any(int(texture_id) >= 0 for texture_id in category_model.mat_texid[wood_id]),
            "the oak worktop material has no texture",
        )
        self.assertTrue(
            any(
                int(texture_id) >= 0
                for texture_id in category_model.mat_texid[yellow_id]
            ),
            "the woven mustard material has no texture",
        )
        for material_id in (wood_id, yellow_id):
            self.assertGreater(float(category_model.mat_specular[material_id]), 0.05)
            self.assertGreater(float(category_model.mat_shininess[material_id]), 0.03)
            self.assertGreater(float(category_model.mat_reflectance[material_id]), 0.0)
            self.assertLess(float(category_model.mat_reflectance[material_id]), 0.5)

    def test_studio_lights_are_active_directional_and_shadowed(self) -> None:
        model = self.category[0]
        benchmark_lights: dict[str, int] = {}
        for light_id in range(model.nlight):
            name = self.mujoco.mj_id2name(
                model,
                self.mujoco.mjtObj.mjOBJ_LIGHT,
                light_id,
            )
            if name and name.startswith("benchmark_"):
                benchmark_lights[str(name)] = light_id

        self.assertTrue(
            {
                "benchmark_key_light",
                "benchmark_fill_light",
                "benchmark_rim_light",
            }
            <= benchmark_lights.keys()
        )
        for light_id in benchmark_lights.values():
            self.assertEqual(int(model.light_active[light_id]), 1)
            self.assertGreater(float(self.np.linalg.norm(model.light_diffuse[light_id])), 0.1)

        key_id = benchmark_lights["benchmark_key_light"]
        self.assertEqual(int(model.light_castshadow[key_id]), 1)
        directions = [
            model.light_dir[light_id]
            / self.np.linalg.norm(model.light_dir[light_id])
            for light_id in benchmark_lights.values()
        ]
        pairwise_alignment = [
            abs(float(self.np.dot(left, right)))
            for index, left in enumerate(directions)
            for right in directions[index + 1 :]
        ]
        self.assertLess(
            min(pairwise_alignment),
            0.95,
            "all studio lights have become effectively collinear",
        )
        self.assertTrue(
            all(float(value) > 0.0 for value in model.vis.headlight.specular),
            "the camera headlight no longer produces material highlights",
        )

    def test_objects_have_semantic_multi_part_geometry(self) -> None:
        category_model = self.category[0]
        place_model = self.place[0]
        category_names = self._geom_names(category_model)
        place_names = self._geom_names(place_model)
        category_by_type = {
            item.object_type: item.object_id
            for item in self.category_scenario.initial_objects
        }
        place_by_type = {
            item.object_type: item.object_id
            for item in self.place_scenario.initial_objects
        }

        expected_category = {
            f"{category_by_type['laptop']}_{suffix}"
            for suffix in ("base", "keyboard", "lid_shell", "screen", "hinge")
        }
        for object_type in ("book", "magazine"):
            expected_category.update(
                f"{category_by_type[object_type]}_{suffix}"
                for suffix in ("cover", "pages", "spine")
            )
        expected_category.update(
            f"{category_by_type['tablet']}_{suffix}"
            for suffix in ("bezel", "screen")
        )

        expected_place = {
            f"{place_by_type['plate']}_{suffix}"
            for suffix in ("rim", "well")
        }
        expected_place.update(
            f"{place_by_type['cup']}_{suffix}"
            for suffix in (
                "body",
                "rim",
                "opening",
                "handle_outer",
                "handle_curve",
                "handle_inner",
            )
        )
        expected_place.update(
            f"{place_by_type['knife']}_{suffix}"
            for suffix in ("blade", "handle", "blade_tip")
        )
        expected_place.update(
            f"{place_by_type['fork']}_{suffix}"
            for suffix in ("handle", "head", "tine_1", "tine_2", "tine_3", "tine_4")
        )

        self.assertFalse(
            expected_category - category_names,
            f"missing category-object geometry: {expected_category - category_names}",
        )
        self.assertFalse(
            expected_place - place_names,
            f"missing place-setting geometry: {expected_place - place_names}",
        )
        for name in expected_category:
            self._material_id_for_geom(category_model, name)
        for name in expected_place:
            self._material_id_for_geom(place_model, name)

    def test_workspace_is_not_clipped_and_retains_surface_detail(self) -> None:
        cases = (
            (
                "category",
                self.category,
                (
                    "benchmark_table_surface",
                    "left_region_geom",
                    "right_region_geom",
                ),
            ),
            (
                "place setting",
                self.place,
                ("benchmark_table_surface", "placemat_geom"),
            ),
        )
        luminance_weights = self.np.asarray((0.2126, 0.7152, 0.0722))
        for case_name, (model, rgb, segmentation), geom_names in cases:
            for geom_name in geom_names:
                with self.subTest(case=case_name, geom=geom_name):
                    geom_id = self._geom_id(model, geom_name)
                    pixels = rgb[segmentation[:, :, 0] == geom_id]
                    self.assertGreater(
                        len(pixels),
                        1_000,
                        f"{geom_name} is not meaningfully visible",
                    )
                    near_white_fraction = float(
                        self.np.mean(self.np.all(pixels >= 245, axis=1))
                    )
                    self.assertLess(
                        near_white_fraction,
                        0.08,
                        f"{geom_name} is overexposed",
                    )
                    luminance = pixels @ luminance_weights
                    dynamic_range = float(
                        self.np.percentile(luminance, 95)
                        - self.np.percentile(luminance, 5)
                    )
                    self.assertGreater(
                        dynamic_range,
                        6.0,
                        f"{geom_name} has lost texture and lighting variation",
                    )

    def test_items_have_clear_boundary_contrast_against_the_workspace(self) -> None:
        cases = (
            (
                self.category_scenario,
                self.category,
                (
                    "benchmark_table_surface",
                    "left_region_geom",
                    "right_region_geom",
                    "left_region_border",
                    "right_region_border",
                ),
            ),
            (
                self.place_scenario,
                self.place,
                (
                    "benchmark_table_surface",
                    "placemat_geom",
                    "placemat_border",
                ),
            ),
        )
        for scenario, (model, rgb, segmentation), workspace_names in cases:
            workspace_ids = [
                self._geom_id(model, name) for name in workspace_names
            ]
            for item in scenario.initial_objects:
                with self.subTest(family=scenario.family, object=item.object_type):
                    object_mask = self._object_mask(
                        model,
                        segmentation,
                        item.object_id,
                    )
                    mask_image = Image.fromarray(
                        object_mask.astype("uint8") * 255,
                    )
                    dilated = (
                        self.np.asarray(mask_image.filter(ImageFilter.MaxFilter(11)))
                        > 0
                    )
                    eroded = (
                        self.np.asarray(mask_image.filter(ImageFilter.MinFilter(7)))
                        > 0
                    )
                    inner_boundary = object_mask & ~eroded
                    exterior_ring = (
                        dilated
                        & ~object_mask
                        & self.np.isin(segmentation[:, :, 0], workspace_ids)
                    )
                    self.assertGreater(int(inner_boundary.sum()), 40)
                    self.assertGreater(int(exterior_ring.sum()), 100)

                    object_colour = self.np.median(
                        rgb[inner_boundary],
                        axis=0,
                    ).astype(float)
                    background_colour = self.np.median(
                        rgb[exterior_ring],
                        axis=0,
                    ).astype(float)
                    colour_distance = float(
                        self.np.linalg.norm(object_colour - background_colour)
                    )
                    self.assertGreater(
                        colour_distance,
                        45.0,
                        f"{item.object_type} blends into its local workspace",
                    )

    def test_blocks_retain_face_shading_and_are_not_flat_colour(self) -> None:
        model, rgb, segmentation = self.blocks
        luminance_weights = self.np.asarray((0.2126, 0.7152, 0.0722))
        for block in self.block_scenario.initial_objects:
            with self.subTest(block=block.object_id):
                pixels = rgb[
                    self._object_mask(model, segmentation, block.object_id)
                ]
                self.assertGreater(len(pixels), 300)
                luminance = pixels @ luminance_weights
                face_range = float(
                    self.np.percentile(luminance, 90)
                    - self.np.percentile(luminance, 10)
                )
                self.assertGreater(
                    face_range,
                    20.0,
                    "block lighting has regressed to a flat 2-D fill",
                )
                self.assertLess(
                    float(self.np.mean(self.np.all(pixels >= 245, axis=1))),
                    0.2,
                    "block faces are clipped by the studio lights",
                )


if __name__ == "__main__":
    unittest.main()
