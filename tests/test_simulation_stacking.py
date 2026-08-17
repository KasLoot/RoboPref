from __future__ import annotations

import threading
import time
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from prefmem.execution.contracts import AnchorKind, ManipulationProgram, ObjectReference
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import SamDetection
from simulation.controller import PandaPickPlaceController
from simulation.stacking import StackingEnvironment


def project_world(calibration, point_world: np.ndarray) -> tuple[float, float, float]:
    rotation = calibration.world_from_camera[:3, :3]
    translation = calibration.world_from_camera[:3, 3]
    point_camera = rotation.T @ (np.asarray(point_world) - translation)
    depth = -float(point_camera[2])
    u = calibration.intrinsic[0, 2] + (
        float(point_camera[0]) * calibration.intrinsic[0, 0] / depth
    )
    v = calibration.intrinsic[1, 2] - (
        float(point_camera[1]) * calibration.intrinsic[1, 1] / depth
    )
    return u, v, depth


class ColorMaskDetector:
    def detect(self, rgb: np.ndarray, prompt: str, *, threshold=None):
        image = rgb.astype(np.int16)
        color = prompt.split()[0].casefold()
        if color == "red":
            mask = (image[:, :, 0] > 100) & (
                image[:, :, 0] > image[:, :, 1] * 3 // 2
            ) & (image[:, :, 0] > image[:, :, 2] * 3 // 2)
        elif color == "green":
            mask = (image[:, :, 1] > 100) & (
                image[:, :, 1] > image[:, :, 0] * 3 // 2
            ) & (image[:, :, 1] > image[:, :, 2] * 3 // 2)
        elif color == "blue":
            mask = (image[:, :, 2] > 100) & (
                image[:, :, 2] > image[:, :, 0] * 3 // 2
            ) & (image[:, :, 2] > image[:, :, 1] * 3 // 2)
        elif color == "white":
            mask = np.all(image > 230, axis=2) & (
                np.max(image, axis=2) - np.min(image, axis=2) < 30
            )
        else:
            return ()
        rows, columns = np.nonzero(mask)
        if not len(rows):
            return ()
        return (
            SamDetection(
                object_id=0,
                box_xyxy=(
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                ),
                mask_area=int(mask.sum()),
                mask=mask,
                score=1.0,
            ),
        )


def program() -> ManipulationProgram:
    return ManipulationProgram.from_dict(
        {
            "schema_version": 1,
            "skill": "pick_place",
            "source": {"query": "red block", "anchor": "top_center"},
            "target": {"query": "white mat", "anchor": "surface_center"},
            "relation": "on_top",
            "orientation": "tool_down",
            "waypoints": [
                {"kind": "approach_source", "clearance_m": 0.1, "gripper": "open"},
                {"kind": "grasp_source", "clearance_m": 0, "gripper": "close"},
                {"kind": "lift", "clearance_m": 0.12, "gripper": "hold"},
                {"kind": "approach_target", "clearance_m": 0.12, "gripper": "hold"},
                {"kind": "place_target", "clearance_m": 0, "gripper": "open"},
                {"kind": "retreat", "clearance_m": 0.12, "gripper": "hold"},
            ],
        }
    )


class StackingSimulationTests(unittest.TestCase):
    def setUp(self):
        self.environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            seed=7,
        )
        self.addCleanup(self.environment.close)

    def test_scene_places_task_blocks_in_view_and_items_outside_reach(self):
        self.assertEqual(self.environment.viewer_camera, "overview")
        snapshot = self.environment.snapshot()
        for name in ("red_block", "green_block", "blue_block"):
            x, y, z = snapshot.object_positions[name]
            self.assertGreaterEqual(x, 0.28)
            self.assertLessEqual(x, 0.80)
            self.assertLessEqual(abs(y), 0.32)
            self.assertAlmostEqual(z, 0.025, delta=0.004)
        for name in ("yellow_block", "purple_block", "orange_block"):
            position = snapshot.object_positions[name]
            self.assertGreater(np.linalg.norm(position[:2]), 1.0)
        current = self.environment.rgbd_frame()
        self.assertEqual(current.rgb.shape, (128, 128, 3))
        self.assertEqual(current.depth_m.shape, (128, 128))
        np.testing.assert_allclose(
            current.calibration.world_from_camera[:3, 3], [1.15, -0.90, 0.90]
        )
        expected_channel = {"red_block": 0, "green_block": 1, "blue_block": 2}
        calibration = current.calibration
        for x in (0.28, 0.80):
            for y in (-0.32, 0.32):
                u, v, depth = project_world(
                    calibration,
                    np.array([x, y, 0.05]),
                )
                self.assertGreater(depth, 0)
                self.assertTrue(0 <= u < current.rgb.shape[1])
                self.assertTrue(0 <= v < current.rgb.shape[0])
        for name in ("yellow_block", "purple_block", "orange_block"):
            u, v, depth = project_world(
                calibration,
                snapshot.object_positions[name],
            )
            self.assertGreater(depth, 0)
            self.assertFalse(
                0 <= u < current.rgb.shape[1]
                and 0 <= v < current.rgb.shape[0]
            )
        for name, channel in expected_channel.items():
            position = snapshot.object_positions[name]
            u_float, v_float, _depth = project_world(
                calibration,
                np.array([position[0], position[1], position[2] + 0.025]),
            )
            u, v = int(round(u_float)), int(round(v_float))
            self.assertTrue(0 <= u < current.rgb.shape[1])
            self.assertTrue(0 <= v < current.rgb.shape[0])
            pixel = current.rgb[v, u]
            self.assertEqual(int(np.argmax(pixel)), channel)
            grounded_center = calibration.backproject(
                np.array([[u, v]], dtype=float),
                np.array([current.depth_m[v, u]], dtype=float),
            )[0]
            self.assertLess(np.linalg.norm(grounded_center[:2] - position[:2]), 0.03)
            self.assertGreaterEqual(grounded_center[2], position[2] - 0.026)
            self.assertLessEqual(grounded_center[2], position[2] + 0.032)

    def test_default_interactive_view_is_free_and_keeps_robot_meshes(self):
        viewer = SimpleNamespace(
            cam=SimpleNamespace(
                type=mujoco.mjtCamera.mjCAMERA_FIXED,
                fixedcamid=self.environment._camera_id,
                lookat=np.zeros(3),
                distance=0.0,
                azimuth=0.0,
                elevation=0.0,
            ),
            opt=SimpleNamespace(geomgroup=np.ones(6, dtype=np.uint8)),
        )

        self.environment._configure_viewer(viewer)

        self.assertEqual(viewer.cam.type, mujoco.mjtCamera.mjCAMERA_FREE)
        self.assertEqual(viewer.cam.fixedcamid, -1)
        self.assertEqual(viewer.opt.geomgroup[2], 1)

    def test_oblique_rgbd_grounding_recovers_cube_tops_and_mat_center(self):
        frame = self.environment.rgbd_frame()
        snapshot = self.environment.snapshot()
        grounder = RGBDGrounder(ColorMaskDetector())
        for color in ("red", "green", "blue"):
            name = f"{color}_block"
            grounded = grounder.ground(
                frame,
                ObjectReference(f"{color} block", AnchorKind.TOP_CENTER),
            )
            expected = snapshot.object_positions[name].copy()
            expected[2] += 0.025
            np.testing.assert_allclose(grounded.point_world, expected, atol=0.012)
        mat = grounder.ground(
            frame,
            ObjectReference("white mat", AnchorKind.SURFACE_CENTER),
        )
        np.testing.assert_allclose(mat.point_world, [0.55, 0.0, 0.012], atol=0.025)

    def test_oblique_camera_exposes_rgb_stack_order(self):
        center = np.array([0.55, 0.0])
        self.environment.place_item("red_block", (*center, 0.037))
        self.environment.place_item("green_block", (*center, 0.087))
        self.environment.place_item("blue_block", (*center, 0.137))
        previous = self.environment.rgbd_frame().sequence
        frame = self.environment.wait_for_frame(
            after_sequence=previous,
            timeout=2.0,
        )
        rgb = frame.rgb.astype(np.int16)
        masks = {
            "red": (rgb[:, :, 0] > 120)
            & (rgb[:, :, 0] > rgb[:, :, 1] * 3 // 2)
            & (rgb[:, :, 0] > rgb[:, :, 2] * 3 // 2),
            "green": (rgb[:, :, 1] > 120)
            & (rgb[:, :, 1] > rgb[:, :, 0] * 3 // 2)
            & (rgb[:, :, 1] > rgb[:, :, 2] * 3 // 2),
            "blue": (rgb[:, :, 2] > 120)
            & (rgb[:, :, 2] > rgb[:, :, 0] * 3 // 2)
            & (rgb[:, :, 2] > rgb[:, :, 1] * 3 // 2),
        }
        median_rows = {}
        for color, mask in masks.items():
            rows, _columns = np.nonzero(mask)
            self.assertGreater(len(rows), 8, f"{color} stack face is not visible")
            median_rows[color] = float(np.median(rows))
        self.assertGreater(median_rows["red"], median_rows["green"])
        self.assertGreater(median_rows["green"], median_rows["blue"])

    def test_cli_default_640_square_framebuffer_renders(self):
        environment = StackingEnvironment(
            viewer=False,
            width=640,
            height=640,
            seed=11,
        )
        try:
            frame = environment.rgbd_frame()
            self.assertEqual(frame.rgb.shape, (640, 640, 3))
            self.assertEqual(frame.depth_m.shape, (640, 640))
        finally:
            environment.close()

    def test_ik_reaches_low_and_clearance_destinations(self):
        controller = PandaPickPlaceController(self.environment)
        seed = controller.solve_ik(np.array([0.55, 0.0, 0.15]))
        target = controller.solve_ik(np.array([0.55, 0.0, 0.039]), seed=seed)
        self.assertEqual(target.shape, (7,))
        self.assertTrue(np.all(np.isfinite(target)))

    def test_controller_respects_speed_and_settles_without_overshoot(self):
        controller = PandaPickPlaceController(self.environment)
        start = self.environment.arm_qpos()
        target_position = np.array([0.50, 0.25, 0.15])
        target = controller.solve_ik(target_position)
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        stop_sampling = threading.Event()

        def sample_arm() -> None:
            while not stop_sampling.is_set():
                samples.append(self.environment.arm_state())
                time.sleep(0.002)

        sampler = threading.Thread(target=sample_arm, daemon=True)
        sampler.start()
        try:
            controller._move_arm(target, threading.Event())
            # Keep sampling after arrival to catch delayed crossings.
            time.sleep(0.20)
        finally:
            stop_sampling.set()
            sampler.join(timeout=1.0)

        positions = np.asarray([sample[0] for sample in samples])
        velocities = np.asarray([sample[1] for sample in samples])
        delta = target - start
        moving = np.abs(delta) > 0.02
        signed_overshoot = np.maximum(
            0.0,
            np.max(
                (positions[:, moving] - target[moving])
                * np.sign(delta[moving]),
                axis=0,
            ),
        )

        self.assertLessEqual(float(np.max(signed_overshoot)), 0.01)
        self.assertLessEqual(float(np.max(np.abs(velocities))), 0.70)
        final_qpos, final_qvel = self.environment.arm_state()
        self.assertLessEqual(
            float(np.max(np.abs(final_qpos - target))),
            controller.settle_position_tolerance_radians,
        )
        self.assertLessEqual(
            float(np.max(np.abs(final_qvel))),
            controller.settle_velocity_tolerance_radians,
        )
        with self.environment._lock:
            final_position = self.environment.data.site_xpos[
                controller._site_id
            ].copy()
        self.assertLess(np.linalg.norm(final_position - target_position), 0.004)

    def test_motion_duration_enforces_peak_speed_and_acceleration(self):
        controller = PandaPickPlaceController(
            self.environment,
            joint_speed_radians=0.6,
            joint_acceleration_radians=1.2,
        )

        duration = controller._motion_duration(0.6)

        self.assertGreaterEqual(duration, 1.25 * 1.875)
        self.assertGreaterEqual(
            duration,
            1.25 * np.sqrt((10.0 / np.sqrt(3.0)) * 0.6 / 1.2),
        )

    def test_controller_physically_builds_a_three_cube_rgb_stack(self):
        self.environment.place_item("red_block", (0.34, -0.26, 0.026))
        self.environment.place_item("green_block", (0.34, 0.26, 0.026))
        self.environment.place_item("blue_block", (0.72, 0.26, 0.026))
        controller = PandaPickPlaceController(
            self.environment,
            joint_speed_radians=1.2,
            joint_acceleration_radians=3.0,
        )
        controller.execute_pick_place(
            program(),
            np.array([0.34, -0.26, 0.05]),
            np.array([0.55, 0.0, 0.012]),
            threading.Event(),
        )
        red = self.environment.snapshot().object_positions["red_block"]
        self.assertLess(np.linalg.norm(red[:2] - np.array([0.55, 0.0])), 0.06)
        self.assertAlmostEqual(red[2], 0.037, delta=0.008)

        second_payload = program().to_dict()
        second_payload["source"] = {
            "query": "green block",
            "anchor": "top_center",
        }
        second_payload["target"] = {
            "query": "red block",
            "anchor": "top_center",
        }
        controller.execute_pick_place(
            ManipulationProgram.from_dict(second_payload),
            np.array([0.34, 0.26, 0.05]),
            red + np.array([0.0, 0.0, 0.025]),
            threading.Event(),
        )
        snapshot = self.environment.snapshot().object_positions
        red = snapshot["red_block"]
        green = snapshot["green_block"]
        self.assertLess(np.linalg.norm(green[:2] - red[:2]), 0.02)
        self.assertAlmostEqual(green[2] - red[2], 0.05, delta=0.008)

        third_payload = program().to_dict()
        third_payload["source"] = {
            "query": "blue block",
            "anchor": "top_center",
        }
        third_payload["target"] = {
            "query": "green block",
            "anchor": "top_center",
        }
        controller.execute_pick_place(
            ManipulationProgram.from_dict(third_payload),
            np.array([0.72, 0.26, 0.05]),
            green + np.array([0.0, 0.0, 0.025]),
            threading.Event(),
        )
        snapshot = self.environment.snapshot().object_positions
        green = snapshot["green_block"]
        blue = snapshot["blue_block"]
        self.assertLess(np.linalg.norm(blue[:2] - green[:2]), 0.02)
        self.assertAlmostEqual(blue[2] - green[2], 0.05, delta=0.008)


if __name__ == "__main__":
    unittest.main()
