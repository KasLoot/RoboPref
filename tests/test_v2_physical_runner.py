from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

import mujoco
import numpy as np

from experiments.harness.video import TerminalCapture, VideoMetadata
from experiments_suite_v2.preflight import SCENE_PATH
from experiments_suite_v2.io import sha256_file
from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.runners.physical import (
    EVIDENCE_FPS,
    HiddenPlacementTruth,
    PHYSICAL_BOARDS,
    PHYSICAL_CUBES,
    PHYSICAL_TARGETS,
    PhysicalProfileCallbacks,
    ProductionPhysicalPath,
    ProfileCallbackError,
    TargetOffsetControllerAdapter,
    V2EvidenceRecorder,
    aggregate_cal_x04,
    aggregate_cal_x05,
    build_balanced_physical_schedule,
    build_cal_x04_rows,
    build_cal_x05_rows,
    build_full_path_task,
    canonical_pick_place_program,
    run_pf_log_readiness_precheck,
    run_oracle_controller_path,
)
from prefmem.execution.frames import CameraCalibration, RGBDFrame
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import SamDetection
from simulation.stacking import StackingEnvironment


def truth_for(row) -> HiddenPlacementTruth:
    target = np.asarray(row.target_surface_world_m, dtype=np.float64)
    source_center = np.array([0.32, -0.34, 0.026], dtype=np.float64)
    return HiddenPlacementTruth(
        row_id=row.row_id,
        cube_body_name=row.cube_body_name,
        source_cube_center_world_m=source_center,
        source_anchor_world_m=source_center + np.array([0.0, 0.0, 0.025]),
        target_surface_anchor_world_m=target,
        expected_settled_cube_center_world_m=target + np.array([0.0, 0.0, 0.025]),
    )


class TraceController:
    def __init__(self, settled: np.ndarray) -> None:
        self.settled = np.asarray(settled, dtype=np.float64)
        self.callback = None
        self.calls = []
        self.holds = 0

    def set_trace_callback(self, callback) -> None:
        self.callback = callback

    def emit(self, kind: str, **payload) -> None:
        if self.callback is not None:
            self.callback({"schema_version": 1, "kind": kind, **payload})

    def execute_pick_place(self, program, source, target, cancel_event) -> None:
        del cancel_event
        source = np.asarray(source, dtype=np.float64).copy()
        target = np.asarray(target, dtype=np.float64).copy()
        self.calls.append((program, source, target))
        object_pre_release = self.settled - np.array([0.001, -0.002, 0.0])
        object_post_open = self.settled - np.array([0.0005, -0.001, 0.0])
        common = lambda value: {
            "object_poses": {
                program.source.query.replace(" cube", "_block"): {
                    "position_world_m": value.tolist()
                }
            }
        }
        commanded = target + np.array([0.0, 0.0, 0.027])
        self.emit(
            "WAYPOINT_ACHIEVED",
            waypoint="place_target",
            commanded_tool_position_world_m=commanded.tolist(),
            tool_position_world_m=(commanded + np.array([0.001, -0.002, 0.0])).tolist(),
        )
        self.emit(
            "OBJECT_AFTER_LIFT",
            tool_position_world_m=(source + np.array([0.0, 0.0, 0.08])).tolist(),
            **common(source + np.array([0.0, 0.0, 0.05])),
        )
        self.emit("OBJECT_PRE_RELEASE", **common(object_pre_release))
        self.emit("OBJECT_POST_OPEN", **common(object_post_open))
        self.emit("OBJECT_POST_RETREAT", **common(self.settled))
        self.emit("OBJECT_SETTLED", **common(self.settled))
        self.emit("PICK_PLACE_COMPLETED", **common(self.settled))

    def safe_hold(self) -> None:
        self.holds += 1


class FixedCompiler:
    def __init__(self, program) -> None:
        self.program = program
        self.tasks = []

    def compile(self, task):
        self.tasks.append(task)
        return self.program


class TwoMaskDetector:
    def __init__(self, source_prompt: str, target_prompt: str) -> None:
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt

    def detect(self, rgb, prompt, *, threshold=None):
        del threshold
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if prompt == self.source_prompt:
            mask[3:10, 3:10] = True
        elif prompt == self.target_prompt:
            mask[18:29, 18:29] = True
        else:
            return ()
        rows, columns = np.nonzero(mask)
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
                score=0.99,
            ),
        )


def production_frame() -> RGBDFrame:
    calibration = CameraCalibration(
        width=32,
        height=32,
        intrinsic=np.array(
            [[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        world_from_camera=np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    return RGBDFrame(
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        depth_m=np.full((32, 32), 0.8, dtype=np.float32),
        calibration=calibration,
        observed_at=1.0,
        sequence=1,
        simulation_time=0.0,
    )


class PhysicalGridTests(unittest.TestCase):
    def test_machine_readable_fixture_matches_runner_constants(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "experiments_suite_v2"
            / "protocol"
            / "physical_cases.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["ordering"]["seed"], 7_853_981)
        self.assertEqual(
            [item["cube_id"] for item in fixture["factors"]["cubes"]],
            [item.cube_id for item in PHYSICAL_CUBES],
        )
        self.assertEqual(
            [item["board_id"] for item in fixture["factors"]["boards"]],
            [item.board_id for item in PHYSICAL_BOARDS],
        )
        self.assertEqual(
            [tuple(item["offset_xy_m"]) for item in fixture["factors"]["targets"]],
            [item.offset_xy_m for item in PHYSICAL_TARGETS],
        )
        self.assertEqual(fixture["aims"]["CAL-X-04"]["row_count"], 300)
        self.assertEqual(fixture["aims"]["CAL-X-05"]["row_count"], 300)

    def test_both_registered_grids_are_exactly_300_rows(self) -> None:
        for rows, aim_id in (
            (build_cal_x04_rows(), "CAL-X-04"),
            (build_cal_x05_rows(), "CAL-X-05"),
        ):
            with self.subTest(aim_id=aim_id):
                self.assertEqual(len(rows), 300)
                self.assertEqual(len({row.row_id for row in rows}), 300)
                self.assertEqual(
                    {row.order_index for row in rows}, set(range(300))
                )
                self.assertEqual(
                    {row.cube_id for row in rows},
                    {cube.cube_id for cube in PHYSICAL_CUBES},
                )
                self.assertEqual(
                    {row.board_id for row in rows},
                    {board.board_id for board in PHYSICAL_BOARDS},
                )
                self.assertEqual(
                    {row.target_id for row in rows},
                    {target.target_id for target in PHYSICAL_TARGETS},
                )
                counts = Counter(
                    (row.cube_id, row.board_id, row.target_id) for row in rows
                )
                self.assertEqual(len(counts), 6 * 2 * 5)
                self.assertEqual(set(counts.values()), {5})
                for repeat in range(5):
                    stratum = [row for row in rows if row.repeat_index == repeat]
                    self.assertEqual(len(stratum), 60)
                    self.assertEqual(
                        len(
                            {
                                (row.cube_id, row.board_id, row.target_id)
                                for row in stratum
                            }
                        ),
                        60,
                    )
                # Five target levels rotate through the same SHA-ordered
                # temporal slots, so x/y location cannot align with wear.
                for within_repeat_index in range(60):
                    self.assertEqual(
                        len(
                            {
                                rows[repeat * 60 + within_repeat_index].target_id
                                for repeat in range(5)
                            }
                        ),
                        5,
                    )

    def test_target_points_are_center_and_four_exact_corners(self) -> None:
        rows = build_cal_x04_rows()
        for board in PHYSICAL_BOARDS:
            offsets = {
                row.target_offset_xy_m
                for row in rows
                if row.board_id == board.board_id
            }
            self.assertEqual(
                offsets,
                {(0.0, 0.0), (-0.06, -0.04), (-0.06, 0.04), (0.06, -0.04), (0.06, 0.04)},
            )
            for row in rows:
                if row.board_id != board.board_id:
                    continue
                self.assertAlmostEqual(
                    row.target_surface_world_m[0],
                    board.center_xy_m[0] + row.target_offset_xy_m[0],
                )
                self.assertAlmostEqual(
                    row.target_surface_world_m[1],
                    board.center_xy_m[1] + row.target_offset_xy_m[1],
                )

    def test_paired_schedule_balances_condition_precedence(self) -> None:
        schedule = build_balanced_physical_schedule()
        self.assertEqual(len(schedule), 600)
        self.assertEqual(
            Counter(entry.row.aim_id for entry in schedule),
            {"CAL-X-04": 300, "CAL-X-05": 300},
        )
        for repeat in range(5):
            for condition_position in (0, 1):
                counts = Counter(
                    entry.row.aim_id
                    for entry in schedule
                    if entry.row.repeat_index == repeat
                    and entry.condition_position == condition_position
                )
                self.assertEqual(counts, {"CAL-X-04": 30, "CAL-X-05": 30})
        self.assertEqual(
            [entry.row.aim_id for entry in schedule[:12]],
            [
                "CAL-X-04", "CAL-X-05", "CAL-X-05", "CAL-X-04",
                "CAL-X-04", "CAL-X-05", "CAL-X-05", "CAL-X-04",
                "CAL-X-04", "CAL-X-05", "CAL-X-05", "CAL-X-04",
            ],
        )
        self.assertEqual(
            [row.to_dict() for row in build_cal_x04_rows()],
            [row.to_dict() for row in build_cal_x04_rows()],
        )

    def test_behavior_critical_fixture_and_runner_hashes_are_unchanged(self) -> None:
        self.assertEqual(
            sha256_file(SUITE_ROOT / "protocol" / "physical_cases.json"),
            "850a01e65702ecc527d4e70cb539f87149a9028e32ba1271b828a10f59bfd42f",
        )
        self.assertEqual(
            sha256_file(SUITE_ROOT / "runners" / "physical.py"),
            "031b9659c550fe4121c3020721ddbf969873fddb4df444ded8748f1440599532",
        )


class PhysicalExecutionSeamTests(unittest.TestCase):
    def test_oracle_path_passes_exact_truth_only_to_controller(self) -> None:
        row = build_cal_x04_rows()[0]
        truth = truth_for(row)
        settled = truth.expected_settled_cube_center_world_m + np.array(
            [0.002, -0.001, 0.0]
        )
        controller = TraceController(settled)
        persisted = []
        evaluation = run_oracle_controller_path(
            row,
            controller=controller,
            truth=truth,
            settled_center_reader=lambda: settled,
            callbacks=PhysicalProfileCallbacks(execution_trace=persisted.append),
        )
        self.assertEqual(len(controller.calls), 1)
        program, source, target = controller.calls[0]
        self.assertEqual(program, canonical_pick_place_program(row))
        np.testing.assert_array_equal(source, truth.source_anchor_world_m)
        np.testing.assert_array_equal(target, truth.target_surface_anchor_world_m)
        np.testing.assert_allclose(
            evaluation.scoring["final_placement_error_mm"], [2.0, -1.0, 0.0]
        )
        self.assertEqual(evaluation.execution_path, "ORACLE_CONTROLLER")
        self.assertTrue(
            evaluation.scoring["scoring_only_truth"]["scoring_only"]
        )
        self.assertEqual(persisted[0]["kind"], "ORACLE_COORDINATES_BOUND")

    def test_profile_callback_failure_is_not_silently_accepted(self) -> None:
        row = build_cal_x04_rows()[0]
        truth = truth_for(row)
        controller = TraceController(truth.expected_settled_cube_center_world_m)

        def broken(_event):
            raise OSError("disk full")

        with self.assertRaises(ProfileCallbackError):
            run_oracle_controller_path(
                row,
                controller=controller,
                truth=truth,
                settled_center_reader=lambda: truth.expected_settled_cube_center_world_m,
                callbacks=PhysicalProfileCallbacks(execution_trace=broken),
            )

    def test_target_offset_adapter_uses_public_fixture_not_truth(self) -> None:
        row = next(
            row
            for row in build_cal_x05_rows()
            if row.target_offset_xy_m == (0.06, -0.04)
        )
        controller = TraceController(np.array([0.0, 0.0, 0.0]))
        adapter = TargetOffsetControllerAdapter(controller)
        events = []
        adapter.set_trace_callback(events.append)
        adapter.configure(row)
        source = np.array([0.3, 0.1, 0.05])
        grounded_board_center = np.array([0.55, 0.16, 0.012])
        adapter.execute_pick_place(
            canonical_pick_place_program(row),
            source,
            grounded_board_center,
            threading.Event(),
        )
        _program, received_source, received_target = controller.calls[0]
        np.testing.assert_array_equal(received_source, source)
        np.testing.assert_allclose(
            received_target,
            grounded_board_center + np.array([0.06, -0.04, 0.0]),
        )
        offset_event = next(
            event for event in events if event["kind"] == "TARGET_GRID_OFFSET_APPLIED"
        )
        self.assertTrue(offset_event["public_fixture"])
        self.assertNotIn("scoring_only", offset_event)

    def test_full_path_uses_real_execution_service_grounder_and_controller(self) -> None:
        row = next(
            row
            for row in build_cal_x05_rows()
            if row.target_offset_xy_m != (0.0, 0.0)
        )
        truth = truth_for(row)
        controller = TraceController(truth.expected_settled_cube_center_world_m)
        compiler = FixedCompiler(canonical_pick_place_program(row))
        grounder = RGBDGrounder(
            TwoMaskDetector(row.cube_selector, row.board_selector)
        )
        traces = []
        perception = []
        with ProductionPhysicalPath(
            compiler=compiler,
            grounder=grounder,
            controller=controller,
            frame_source=production_frame,
            callbacks=PhysicalProfileCallbacks(
                execution_trace=traces.append,
                perception=perception.append,
            ),
        ) as path:
            task = build_full_path_task(row, published_at=1.0)
            evaluation = path.run(
                row,
                task=task,
                truth=truth,
                settled_center_reader=lambda: truth.expected_settled_cube_center_world_m,
                timeout_seconds=2.0,
            )
        self.assertEqual(evaluation.execution_path, "PRODUCTION_FULL_PATH")
        self.assertEqual(evaluation.scoring["final_radial_xy_error_mm"], 0.0)
        self.assertTrue(evaluation.telemetry["production_program_contract_pass"])
        kinds = {event["kind"] for event in traces}
        self.assertIn("RGBD_FRAME_CAPTURED", kinds)
        self.assertIn("GROUNDING_COMPLETE", kinds)
        self.assertIn("TARGET_GRID_OFFSET_APPLIED", kinds)
        self.assertIn("PICK_PLACE_COMPLETED", kinds)
        self.assertEqual(len(compiler.tasks), 1)
        self.assertFalse(hasattr(compiler.tasks[0], "hidden_truth"))
        self.assertTrue(perception)
        raw_target = next(
            event["grounded_board_anchor_world_m"]
            for event in traces
            if event["kind"] == "TARGET_GRID_OFFSET_APPLIED"
        )
        np.testing.assert_allclose(
            controller.calls[0][2][:2],
            np.asarray(raw_target)[:2] + np.asarray(row.target_offset_xy_m),
        )


class PhysicalGateTests(unittest.TestCase):
    @staticmethod
    def records(rows, error):
        radial = float(np.linalg.norm(np.asarray(error, dtype=float)[:2]))
        return [
            {
                **row.to_dict(),
                "scoring": {
                    "final_placement_error_mm": list(error),
                    "final_radial_xy_error_mm": radial,
                },
            }
            for row in rows
        ]

    def test_controller_gate_uses_frozen_inclusive_limits_and_strata(self) -> None:
        rows = build_cal_x04_rows()
        gate = aggregate_cal_x04(self.records(rows, [3.0, -3.0, 0.0]))
        self.assertEqual(gate["status"], "PASS")
        self.assertTrue(gate["complete_grid"])
        self.assertEqual(set(gate["strata"]["cube_id"]), {
            cube.cube_id for cube in PHYSICAL_CUBES
        })
        self.assertEqual(set(gate["strata"]["board_id"]), {
            board.board_id for board in PHYSICAL_BOARDS
        })

        signed_fail = aggregate_cal_x04(
            self.records(rows, [3.001, 0.0, 0.0])
        )
        self.assertEqual(signed_fail["status"], "FAIL")
        radial_records = []
        for index, row in enumerate(rows):
            x = 9.0 if index % 2 else -9.0
            radial_records.extend(self.records((row,), [x, 0.0, 0.0]))
        radial_fail = aggregate_cal_x04(radial_records)
        self.assertTrue(radial_fail["checks"]["median_signed_x_y_pass"])
        self.assertFalse(radial_fail["checks"]["p95_radial_xy_pass"])
        self.assertEqual(radial_fail["status"], "FAIL")

    def test_full_path_gate_and_incomplete_grid_status(self) -> None:
        rows = build_cal_x05_rows()
        gate = aggregate_cal_x05(self.records(rows, [5.0, -5.0, 0.0]))
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["thresholds"]["p95_radial_xy_limit_mm"], 12.0)
        incomplete = aggregate_cal_x05(self.records(rows[:1], [0.0, 0.0, 0.0]))
        self.assertEqual(incomplete["status"], "INCOMPLETE")


class EvidenceAdapterTests(unittest.TestCase):
    def test_slow_unique_renders_fill_a_monotonic_timeline_and_end_fresh(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        environment = SimpleNamespace(
            model=model,
            data=mujoco.MjData(model),
            _lock=threading.RLock(),
        )
        gradient = np.zeros((960, 960, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.arange(960, dtype=np.uint16)[:, None] % 256
        gradient[:, :, 1] = np.arange(960, dtype=np.uint16)[None, :] % 256
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            terminal = TerminalCapture(attempt / "terminal.log")
            recorder = V2EvidenceRecorder(
                environment,
                terminal,
                attempt,
                controller_only=True,
                final_hold_seconds=0.0,
            )
            render_indices = []

            def slow_render(_renderer, _camera, _scene_option):
                render_indices.append(len(render_indices) + 1)
                time.sleep(0.14)
                result = gradient.copy()
                result[0, 0, 2] = render_indices[-1] % 256
                return result

            recorder._render_simulation = slow_render
            try:
                recorder.start()
                threading.Event().wait(0.35)
                metadata = recorder.stop()
                self.assertTrue(recorder._exact_final_render_completed)
                self.assertGreater(
                    recorder._duplicate_frame_count,
                    0,
                )
                self.assertEqual(
                    metadata.frame_count,
                    recorder._unique_render_count
                    + recorder._duplicate_frame_count,
                )
                self.assertEqual(recorder._unique_render_count, len(render_indices))
                self.assertEqual(
                    recorder._verification["encoded_frame_count"],
                    metadata.frame_count,
                )
                self.assertTrue(
                    recorder._verification["duration_agreement_pass"]
                )
            finally:
                if recorder.running:
                    recorder.stop()
                terminal.close()

    def test_evidence_renderer_copies_state_then_releases_environment_lock(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

        class InspectableLock:
            def __init__(self) -> None:
                self._lock = threading.RLock()
                self.held = False

            def __enter__(self):
                self._lock.acquire()
                self.held = True
                return self

            def __exit__(self, *_args) -> None:
                self.held = False
                self._lock.release()

        lock = InspectableLock()
        environment = SimpleNamespace(
            model=model,
            data=mujoco.MjData(model),
            _lock=lock,
        )

        class InspectingRenderer:
            def __init__(self) -> None:
                self.states = []

            def update_scene(self, data, *, camera, scene_option) -> None:
                self.assertions(data, camera, scene_option)

            def assertions(self, data, camera, scene_option) -> None:
                self_outer.assertFalse(lock.held)
                self_outer.assertIsNot(data, environment.data)
                self_outer.assertEqual(camera.fixedcamid, expected_camera_id)
                self_outer.assertEqual(int(scene_option.geomgroup[2]), 1)
                self_outer.assertEqual(int(scene_option.geomgroup[3]), 0)
                self.states.append(data.qpos.copy())

            def render(self):
                return np.zeros((960, 960, 3), dtype=np.uint8)

        self_outer = self
        expected_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "prefmem_camera"
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            terminal = TerminalCapture(attempt / "terminal.log")
            try:
                recorder = V2EvidenceRecorder(
                    environment,
                    terminal,
                    attempt,
                    controller_only=True,
                    final_hold_seconds=0.0,
                )
                renderer = InspectingRenderer()
                camera = recorder._build_camera()
                option = mujoco.MjvOption()
                with lock:
                    environment.data.qpos[0] = 0.123
                    mujoco.mj_forward(model, environment.data)
                frame = recorder._render_simulation(renderer, camera, option)
                self.assertEqual(frame.shape, (960, 960, 3))
                self.assertAlmostEqual(renderer.states[-1][0], 0.123)
                first_snapshot = recorder._evidence_render_data
                with lock:
                    environment.data.qpos[0] = -0.234
                    mujoco.mj_forward(model, environment.data)
                recorder._render_simulation(renderer, camera, option)
                self.assertIs(recorder._evidence_render_data, first_snapshot)
                self.assertAlmostEqual(renderer.states[-1][0], -0.234)
            finally:
                terminal.close()

    def test_pf_log_readiness_rejects_non_realtime_environment_before_output(self) -> None:
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            viewer=False,
            start=False,
            realtime=False,
            width=64,
            height=64,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "benchmark"
                with self.assertRaisesRegex(
                    RuntimeError, "realtime=True"
                ):
                    run_pf_log_readiness_precheck(
                        environment,
                        destination,
                    )
                self.assertFalse(destination.exists())
        finally:
            environment.close()

    def test_recorder_freezes_approved_camera_separate_renderer_and_portable_paths(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        environment = SimpleNamespace(
            model=model,
            data=mujoco.MjData(model),
            _lock=threading.RLock(),
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            terminal = TerminalCapture(attempt / "terminal.log")
            try:
                recorder = V2EvidenceRecorder(
                    environment,
                    terminal,
                    attempt,
                    controller_only=True,
                    final_hold_seconds=0.0,
                )
                self.assertEqual(recorder.fps, EVIDENCE_FPS)
                self.assertEqual(
                    (recorder.FRAME_WIDTH, recorder.FRAME_HEIGHT), (1920, 1080)
                )
                camera = recorder._build_camera()
                expected_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, "prefmem_camera"
                )
                self.assertEqual(camera.type, mujoco.mjtCamera.mjCAMERA_FIXED)
                self.assertEqual(camera.fixedcamid, expected_id)
                self.assertFalse(recorder.camera_policy["agent_input"])
                self.assertFalse(recorder.camera_policy["sam_input"])
                self.assertIn("SEPARATE_EVIDENCE_ONLY", recorder.camera_policy["renderer"])
                recorder._verification = {"decodable": True}
                metadata = VideoMetadata(
                    path=str(attempt / "video.mp4"),
                    width=1920,
                    height=1080,
                    fps=10.0,
                    frame_count=30,
                    duration_seconds=3.0,
                    file_size_bytes=100,
                    codec="mp4v",
                    initial_png=str(attempt / "initial.png"),
                    final_png=str(attempt / "final.png"),
                )
                portable = recorder.portable_metadata(metadata)
                self.assertEqual(portable["path"], "video.mp4")
                self.assertEqual(portable["initial_png"], "initial.png")
                self.assertEqual(portable["final_png"], "final.png")
                self.assertEqual(portable["artifact_profile"], "FULL_PHYSICAL")
                self.assertIn("PrefMem agents: NOT ACTIVE", terminal.rolling_text())
            finally:
                terminal.close()


if __name__ == "__main__":
    unittest.main()
