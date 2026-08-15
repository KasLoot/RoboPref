from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock

import mujoco
import numpy as np

from experiments_suite_v2.preflight import (
    EXPECTED_BLOCKS,
    SCENE_PATH,
    compile_scene,
    validate_camera_freeze,
    validate_portable_scene_copy,
    validate_scene_source,
    verify_robot_asset_manifest,
)
from prefmem.execution.frames import CameraCalibration
from simulation.stacking import (
    ALL_BLOCKS,
    TB6C_V00_POSITIONS,
    StackingEnvironment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCENE = REPOSITORY_ROOT / "src" / "simulation" / "stacking_scene.xml"


def project_world(
    calibration: CameraCalibration,
    point_world: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
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


class TB6CResetTests(unittest.TestCase):
    def test_v00_is_exact_and_reproducible_for_twenty_resets(self) -> None:
        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            scene_path=SCENE_PATH,
            start=False,
        )
        self.addCleanup(environment.close)
        self.assertEqual(environment.reset_profile, "TB6C_V00")
        expected = dict(TB6C_V00_POSITIONS)

        for reset_index in range(20):
            with self.subTest(reset_index=reset_index):
                environment.reset()
                snapshot = environment.snapshot()
                self.assertEqual(set(snapshot.object_positions), set(ALL_BLOCKS))
                for name, position in expected.items():
                    np.testing.assert_array_equal(
                        snapshot.object_positions[name],
                        np.asarray(position),
                    )

    def test_legacy_scene_keeps_random_and_offscreen_reset_contract(self) -> None:
        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            seed=9,
            scene_path=LEGACY_SCENE,
            start=False,
        )
        self.addCleanup(environment.close)
        self.assertEqual(environment.reset_profile, "LEGACY_RANDOM")
        self.assertEqual(environment._sam_camera_id, environment._camera_id)
        self.assertEqual(environment._prefmem_camera_id, environment._camera_id)
        self.assertIs(environment._task_scene_option, environment._sam_scene_option)
        first = environment.snapshot()
        environment.reset()
        second = environment.snapshot()
        self.assertTrue(
            any(
                not np.array_equal(
                    first.object_positions[name], second.object_positions[name]
                )
                for name in ("red_block", "green_block", "blue_block")
            )
        )
        for name in ("yellow_block", "purple_block", "orange_block"):
            self.assertGreater(
                np.linalg.norm(second.object_positions[name][:2]),
                1.0,
            )


class TB6CDualCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = StackingEnvironment(
            viewer=False,
            width=256,
            height=256,
            render_hz=30.0,
            scene_path=SCENE_PATH,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.environment.close()

    def test_streams_are_synchronized_distinct_and_exactly_calibrated(self) -> None:
        with self.environment._lock:
            sam = self.environment.sam_rgbd_frame()
            prefmem = self.environment.prefmem_rgbd_frame()
        self.assertEqual(sam.sequence, prefmem.sequence)
        self.assertEqual(sam.observed_at, prefmem.observed_at)
        self.assertEqual(sam.simulation_time, prefmem.simulation_time)
        np.testing.assert_allclose(
            sam.calibration.world_from_camera[:3, 3],
            [0.55, 0.0, 1.20],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            prefmem.calibration.world_from_camera[:3, 3],
            [1.50, 0.0, 1.01],
            rtol=0.0,
            atol=1e-12,
        )
        expected_focal = 0.5 * 256 / np.tan(np.radians(48.0) / 2.0)
        self.assertAlmostEqual(sam.calibration.intrinsic[0, 0], expected_focal)
        self.assertAlmostEqual(prefmem.calibration.intrinsic[0, 0], expected_focal)
        self.assertGreater(
            float(np.mean(np.abs(sam.rgb.astype(float) - prefmem.rgb.astype(float)))),
            5.0,
        )

    def test_only_sam_stream_hides_robot_and_sites(self) -> None:
        self.assertEqual(int(self.environment._sam_scene_option.geomgroup[2]), 0)
        self.assertEqual(int(self.environment._sam_scene_option.geomgroup[3]), 0)
        self.assertTrue(np.all(self.environment._sam_scene_option.sitegroup == 0))
        self.assertEqual(int(self.environment._prefmem_scene_option.geomgroup[2]), 1)
        self.assertEqual(int(self.environment._prefmem_scene_option.geomgroup[3]), 0)

    def test_complete_workspace_projects_inside_both_approved_views(self) -> None:
        frames = (
            self.environment.sam_rgbd_frame(),
            self.environment.prefmem_rgbd_frame(),
        )
        board_corners = tuple(
            (x, y, 0.012)
            for x in (0.42, 0.68)
            for y in (-0.26, -0.06, 0.06, 0.26)
        )
        cube_top_corners = tuple(
            (position[0] + dx, position[1] + dy, 0.051)
            for position in EXPECTED_BLOCKS.values()
            for dx in (-0.025, 0.025)
            for dy in (-0.025, 0.025)
        )
        for stream_name, frame in zip(("sam", "prefmem"), frames, strict=True):
            for point in (*board_corners, *cube_top_corners):
                with self.subTest(stream=stream_name, point=point):
                    u, v, depth = project_world(frame.calibration, point)
                    self.assertGreater(depth, 0.0)
                    self.assertGreaterEqual(u, 0.0)
                    self.assertLess(u, frame.rgb.shape[1])
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLess(v, frame.rgb.shape[0])

    def test_legacy_frame_aliases_route_to_the_intended_streams(self) -> None:
        # Hold the environment lock so the render thread cannot publish the
        # next synchronized pair between the alias and canonical reads.
        with self.environment._lock:
            self.assertIs(
                self.environment.rgbd_frame(),
                self.environment.sam_rgbd_frame(),
            )
            legacy_capture = self.environment.captured_frame()
            prefmem_capture = self.environment.prefmem_captured_frame()
        self.assertEqual(legacy_capture.sequence, prefmem_capture.sequence)
        self.assertEqual(
            legacy_capture.image_block,
            prefmem_capture.image_block,
        )

    def test_render_worker_does_not_hold_the_physics_lock(self) -> None:
        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            render_hz=30.0,
            scene_path=SCENE_PATH,
            start=False,
        )
        entered = threading.Event()
        release = threading.Event()
        original = environment._render_snapshot_pair

        def blocking_render(renderer, snapshot):
            if not entered.is_set():
                entered.set()
                if not release.wait(2.0):
                    raise RuntimeError("test did not release render worker")
            return original(renderer, snapshot)

        environment._render_snapshot_pair = blocking_render
        starter = threading.Thread(target=environment.start, daemon=True)
        starter.start()
        try:
            self.assertTrue(entered.wait(2.0))
            started = time.monotonic()
            positions, velocities = environment.arm_state()
            elapsed = time.monotonic() - started
            self.assertEqual(positions.shape, (7,))
            self.assertEqual(velocities.shape, (7,))
            self.assertLess(elapsed, 0.10)
        finally:
            release.set()
            starter.join(timeout=5.0)
            environment.close()
        self.assertFalse(starter.is_alive())

    def test_reset_revision_discards_an_inflight_prereset_pair(self) -> None:
        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            render_hz=30.0,
            scene_path=SCENE_PATH,
            start=False,
        )
        entered = threading.Event()
        release = threading.Event()
        captured_revisions = []
        original = environment._render_snapshot_pair

        def blocking_first_render(renderer, snapshot):
            captured_revisions.append(snapshot.state_revision)
            if len(captured_revisions) == 1:
                entered.set()
                if not release.wait(2.0):
                    raise RuntimeError("test did not release stale render")
            return original(renderer, snapshot)

        environment._render_snapshot_pair = blocking_first_render
        starter = threading.Thread(target=environment.start, daemon=True)
        starter.start()
        try:
            self.assertTrue(entered.wait(2.0))
            stale_revision = captured_revisions[0]
            environment.reset()
            reset_revision = environment._state_revision
            self.assertGreater(reset_revision, stale_revision)
            with self.assertRaisesRegex(RuntimeError, "has not rendered"):
                environment.sam_rgbd_frame()
            release.set()
            starter.join(timeout=5.0)
            self.assertFalse(starter.is_alive())
            sam = environment.sam_rgbd_frame()
            prefmem = environment.prefmem_rgbd_frame()
            self.assertEqual(environment._last_published_state_revision, reset_revision)
            self.assertEqual(sam.sequence, 1)
            self.assertEqual(prefmem.sequence, 1)
            self.assertEqual(sam.observed_at, prefmem.observed_at)
            self.assertEqual(sam.simulation_time, prefmem.simulation_time)
        finally:
            release.set()
            environment.close()

    def test_paused_periodic_stream_allows_an_explicit_synchronized_pair(self) -> None:
        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            render_hz=30.0,
            scene_path=SCENE_PATH,
        )
        try:
            environment.pause_periodic_rendering()
            self.assertTrue(environment.periodic_rendering_paused)
            # Allow an already in-flight render to finish, then confirm no
            # periodic publication advances the stream.
            threading.Event().wait(0.20)
            paused_sequence = environment.sam_rgbd_frame().sequence
            threading.Event().wait(0.20)
            self.assertEqual(
                environment.sam_rgbd_frame().sequence,
                paused_sequence,
            )
            sam, prefmem = environment.render_camera_pair(timeout=2.0)
            self.assertGreater(sam.sequence, paused_sequence)
            self.assertEqual(sam.sequence, prefmem.sequence)
            self.assertEqual(sam.observed_at, prefmem.observed_at)
            self.assertEqual(sam.simulation_time, prefmem.simulation_time)
            self.assertTrue(environment.periodic_rendering_paused)
            environment.resume_periodic_rendering()
            self.assertFalse(environment.periodic_rendering_paused)
            resumed = environment.wait_for_sam_frame(
                after_sequence=sam.sequence,
                timeout=2.0,
            )
            self.assertGreater(resumed.sequence, sam.sequence)
        finally:
            environment.close()


class TB6CAssetAndScenePreflightTests(unittest.TestCase):
    def test_frozen_asset_manifest_and_camera_configuration_verify(self) -> None:
        manifest = verify_robot_asset_manifest()
        self.assertTrue(manifest["robot_asset_id"].startswith("sha256:"))
        self.assertEqual(manifest["published_file_count"], 81)
        self.assertEqual(manifest["runtime_dependency_count"], 70)
        freeze = validate_camera_freeze()
        self.assertEqual(freeze["approval_status"], "APPROVED")

    def test_scene_source_is_canonical_and_compiles(self) -> None:
        validate_scene_source()
        model = compile_scene()
        self.assertGreater(model.nbody, 0)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "sam_camera"),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "prefmem_camera",
            ),
            0,
        )

    def test_scene_compiles_after_the_suite_assets_are_moved(self) -> None:
        validate_portable_scene_copy()


class RuntimeCameraRoutingTests(unittest.TestCase):
    def test_build_runtime_fences_sam_from_prefmem_components(self) -> None:
        from prefmem import runtime as runtime_module

        class FakeEnvironment:
            def sam_rgbd_frame(self):
                raise AssertionError("not called during composition")

            def prefmem_captured_frame(self):
                raise AssertionError("not called during composition")

            def close(self):
                pass

        environment = FakeEnvironment()
        hri = SimpleNamespace(
            planner_agent=object(),
            config=SimpleNamespace(model="frozen-gemma", model_base_url="http://model"),
            metrics=object(),
            attach_runtime=mock.Mock(),
        )
        built_runtime = object()
        arguments = SimpleNamespace(
            model_config=object(),
            executor="mujoco",
            simulation_seed=4,
            simulation_render_size=128,
            simulation_render_hz=5.0,
            simulation_viewer=False,
            simulation_viewer_camera="overview",
            simulation_scene=SCENE_PATH,
            sam_base_url="http://127.0.0.1:9000",
            sam_threshold=0.5,
            execution_model="execution-gemma",
            execution_model_base_url="http://127.0.0.1:8000/v1",
            camera_base_url="http://127.0.0.1:1234",
            monitor_min_interval=1.0,
            success_confirmations=2,
            success_stability_seconds=2.0,
            failure_confirmations=2,
            monitor_timeout=30.0,
            max_planning_cycles=20,
            auto_timeout_replan=True,
            max_consecutive_timeout_replans=2,
            max_timeout_replans_per_instruction=2,
        )
        with (
            mock.patch("prefmem.agents.hri.HRI_Agent", return_value=hri),
            mock.patch(
                "simulation.stacking.StackingEnvironment",
                return_value=environment,
            ),
            mock.patch("prefmem.execution.sam.Sam3Client") as sam_client,
            mock.patch("prefmem.execution.gemma.GemmaExecutionCompiler"),
            mock.patch("prefmem.execution.grounding.RGBDGrounder"),
            mock.patch("simulation.controller.PandaPickPlaceController"),
            mock.patch("prefmem.execution.service.ExecutionService") as service,
            mock.patch.object(
                runtime_module,
                "PrefMemRuntime",
                return_value=built_runtime,
            ) as runtime_factory,
        ):
            result_runtime, result_hri = runtime_module.build_runtime(arguments)

        self.assertIs(result_runtime, built_runtime)
        self.assertIs(result_hri, hri)
        sam_source = service.call_args.args[3]
        self.assertIs(sam_source.__self__, environment)
        self.assertIs(sam_source.__func__, FakeEnvironment.sam_rgbd_frame)
        prefmem_source = runtime_factory.call_args.kwargs["frame_source"]
        self.assertIs(prefmem_source.__self__, environment)
        self.assertIs(
            prefmem_source.__func__,
            FakeEnvironment.prefmem_captured_frame,
        )
        self.assertNotEqual(sam_source.__func__, prefmem_source.__func__)
        self.assertFalse(
            service.call_args.kwargs["enable_oracle_grounding_fallback"]
        )
        self.assertIsNone(
            service.call_args.kwargs["oracle_grounding_provider"]
        )
        self.assertEqual(
            runtime_factory.call_args.kwargs["owned_resources"],
            (sam_client.return_value, environment),
        )
        hri.attach_runtime.assert_called_once_with(built_runtime)


if __name__ == "__main__":
    unittest.main()
