from __future__ import annotations

from collections import deque
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from experiments_suite_v2.audit import AuditReport
from experiments_suite_v2.io import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    iter_jsonl,
    load_json,
    sha256_file,
)
from experiments_suite_v2.preflight import SCENE_PATH
from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.runners.physical import (
    HiddenPlacementTruth,
    PhysicalProfileCallbacks,
    ResetVerificationError,
    build_cal_x04_rows,
    build_cal_x05_rows,
    evaluate_placement,
    run_pf_motion_precheck,
    run_pf_reset_precheck,
    verify_tb6c_reset,
)
from experiments_suite_v2.runners.physical_campaign import (
    BalancedPhysicalCampaignDriver,
    EXPECTED_PHYSICAL_SCENE_SHA256,
    EXPECTED_PHYSICAL_FIXTURE_SHA256,
    EXPECTED_TASK_STREAM_RESOLUTION,
    FullPhysicalEvidenceSession,
    PhysicalCalibrationCampaign,
    PhysicalCampaignError,
    PhysicalCampaignPins,
    PhysicalCampaignRecoveryRequired,
    PhysicalPreflightEvidence,
    PhysicalTaskStreamGateError,
    _BoundedHighRateTraceWriter,
    cal_x04_dry_plan,
    materialize_v2_physical_scene,
    validate_physical_configuration_pins,
    verify_approved_live_task_streams,
    write_physical_preflight_bundle,
)
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    AttemptIdentity,
    FormalVerdict,
    ResultEnvelope,
    TrialTuple,
)
from experiments_suite_v2.services import ServiceGateError
from experiments_suite_v2.storage import RunStore
from prefmem.execution.frames import CameraCalibration, RGBDFrame
from simulation.stacking import StackingEnvironment


_PHYSICAL_PIN_PATHS = (
    "scenes/tb6c_v2.xml",
    "protocol/physical_cases.json",
    "protocol/camera_approval.json",
    "protocol/camera_runtime_freeze.json",
    "protocol/evidence_camera_policy.json",
    "camera_previews/motion_revision_candidate/camera_metadata.json",
)


def _copy_physical_pins(destination: Path) -> None:
    for relative in _PHYSICAL_PIN_PATHS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SUITE_ROOT / relative, target)


def _pf_log_report() -> dict:
    return {
        "schema_version": "prefmem.test-pf-log.v2",
        "analytical_classification": "PASS",
        "formal_verdict": "PASS",
        "must_pass": {
            "physical_execution_completed": True,
            "full_physical_artifact_audit": True,
        },
        "artifact_profile": "FULL_PHYSICAL",
        "attempt_audit_passed": True,
        "realtime": True,
        "task_stream_resolution": list(EXPECTED_TASK_STREAM_RESOLUTION),
        "scene_revision": "TB6C-v2-PF-MOTION-01",
        "scene_sha256": EXPECTED_PHYSICAL_SCENE_SHA256,
    }


def _truth(row) -> HiddenPlacementTruth:
    source_center = np.array([0.32, -0.34, 0.026], dtype=np.float64)
    target = np.asarray(row.target_surface_world_m, dtype=np.float64)
    return HiddenPlacementTruth(
        row_id=row.row_id,
        cube_body_name=row.cube_body_name,
        source_cube_center_world_m=source_center,
        source_anchor_world_m=source_center + np.array([0.0, 0.0, 0.025]),
        target_surface_anchor_world_m=target,
        expected_settled_cube_center_world_m=target
        + np.array([0.0, 0.0, 0.025]),
    )


class FakeStore:
    """Small file-backed RunStore seam; profile bytes are supplied elsewhere."""

    def __init__(self, suite_root: Path, run_id: str, pins: PhysicalCampaignPins):
        self.suite_root = suite_root
        self._run = suite_root / "results/shared-campaigns" / run_id
        self._run.mkdir(parents=True)
        atomic_write_json(
            self._run / "run_manifest.json",
            {
                "suite_run_id": run_id,
                "suite_version": pins.suite_version,
                "source_commit": pins.source_commit,
                "status": "OPEN",
            },
        )

    def run_path(self, _run_id: str) -> Path:
        return self._run

    def allocate_attempt(self, _run_id, trial, case, **_kwargs):
        return self._allocate(trial, case, 0, None, None)

    def allocate_retry(self, previous: Path):
        manifest = load_json(previous / "attempt_manifest.json")
        old = AttemptIdentity.from_dict(manifest["identity"])
        return self._allocate(
            TrialTuple.from_dict(manifest["trial_tuple"]),
            manifest["case_definition"],
            old.attempt_number + 1,
            old.retry_of or old.attempt_id,
            old.attempt_id,
        )

    def _allocate(self, trial, case, number, retry_of, supersedes):
        identity = AttemptIdentity(
            trial_id=trial.trial_id,
            trial_tuple_sha256=trial.trial_tuple_sha256,
            attempt_number=number,
            retry_of=retry_of,
            supersedes_attempt=supersedes,
        )
        path = (
            self._run
            / "CAL-X"
            / f"{trial.case_id}__{trial.variant_id}__A{number}"
        )
        path.mkdir(parents=True, exist_ok=False)
        case_payload = case.to_dict() if hasattr(case, "to_dict") else case
        atomic_write_json(
            path / "attempt_manifest.json",
            {
                "identity": identity.to_dict(),
                "trial_tuple": trial.to_dict(),
                "case_definition": case_payload,
                "status": "ALLOCATED",
            },
        )
        append_jsonl(path / "events.jsonl", {"event_type": "ALLOCATED"})
        return path

    def start_attempt(self, path: Path):
        manifest = load_json(path / "attempt_manifest.json")
        manifest["status"] = "RUNNING"
        atomic_write_json(path / "attempt_manifest.json", manifest)

    def write_result(self, path: Path, result: ResultEnvelope):
        atomic_write_json(path / "results.json", result.to_dict(), overwrite=False)

    def finalize_attempt(self, path: Path):
        manifest = load_json(path / "attempt_manifest.json")
        manifest["status"] = "FINALIZED"
        atomic_write_json(path / "attempt_manifest.json", manifest)
        atomic_write_json(path / "artifact_manifest.json", {"frozen": True})

    def append_event(self, path: Path, event_type: str, payload):
        append_jsonl(
            path / "events.jsonl", {"event_type": event_type, "payload": payload}
        )


class FakeSession:
    def __init__(self, attempt_dir: Path, row, attempt_id: str):
        self.attempt_dir = attempt_dir
        self.row = row
        self.attempt_id = attempt_id
        self.trace_records = []
        self.callbacks = PhysicalProfileCallbacks(
            execution_trace=self.trace_records.append
        )
        self.health = None
        self.finished = None

    def begin(self):
        append_jsonl(self.attempt_dir / "fake_profile.jsonl", {"stage": "BEGIN"})

    def prepare_row(self):
        return _truth(self.row)

    def start_recorder(self):
        append_jsonl(
            self.attempt_dir / "fake_profile.jsonl", {"stage": "RECORDER"}
        )

    def verify_task_stream_gate(self):
        payload = {
            "schema_version": "prefmem.cal-x04-task-stream-gate.v2",
            "status": "PASS",
            "checked_before_motion": True,
            "motion_dispatch_count_at_check": 0,
            "approved_resolution": [768, 768],
        }
        atomic_write_json(
            self.attempt_dir / "task_stream_pre_motion_gate.json",
            payload,
            overwrite=False,
        )
        return payload

    def record_task_stream_gate_failure(self, error):
        payload = {
            "schema_version": "prefmem.cal-x04-task-stream-gate.v2",
            "status": "FAIL",
            "checked_before_motion": True,
            "motion_dispatch_count_at_check": 0,
            "message": str(error),
        }
        atomic_write_json(
            self.attempt_dir / "task_stream_pre_motion_gate.json",
            payload,
            overwrite=False,
        )
        return payload

    def record_task_stream_gate_not_reached(self, *, reason):
        payload = {
            "schema_version": "prefmem.cal-x04-task-stream-gate.v2",
            "status": "NOT_REACHED",
            "checked_before_motion": False,
            "motion_dispatch_count_at_check": 0,
            "reason": reason,
        }
        atomic_write_json(
            self.attempt_dir / "task_stream_pre_motion_gate.json",
            payload,
            overwrite=False,
        )
        return payload

    def record_health(self, records, *, status, failure=None):
        self.health = (status, list(records), failure)
        atomic_write_json(
            self.attempt_dir / "service_health.json",
            {"status": status, "records": list(records), "failure": failure},
        )

    def record_failure(self, stage, error):
        atomic_write_json(
            self.attempt_dir / "failure.json",
            {"stage": stage, "error_type": type(error).__name__},
        )

    def finish(self, result, evaluation):
        self.finished = (result, evaluation)


class FakeSessionFactory:
    def __init__(self):
        self.sessions = []

    def __call__(self, attempt_dir, *, row, suite_run_id, attempt_id):
        del suite_run_id
        session = FakeSession(attempt_dir, row, attempt_id)
        self.sessions.append(session)
        return session


class FailingTaskStreamSession(FakeSession):
    def verify_task_stream_gate(self):
        raise PhysicalTaskStreamGateError("deterministic 640 x 640 mismatch")


class FailingTaskStreamSessionFactory(FakeSessionFactory):
    def __call__(self, attempt_dir, *, row, suite_run_id, attempt_id):
        del suite_run_id
        session = FailingTaskStreamSession(attempt_dir, row, attempt_id)
        self.sessions.append(session)
        return session


class FakeExecutor:
    def __init__(self, execution_path: str, outcomes=()):
        self.execution_path = execution_path
        self.outcomes = deque(outcomes)
        self.calls = []

    def execute(self, row, truth, callbacks):
        self.calls.append(row.row_id)
        if self.outcomes and self.outcomes.popleft() == "FAIL":
            if callbacks.execution_trace:
                callbacks.execution_trace(
                    {"kind": "MOTION_FAILED", "row_id": row.row_id}
                )
            raise RuntimeError("deterministic controller failure")
        return evaluate_placement(
            row,
            execution_path=self.execution_path,
            truth=truth,
            settled_center_world_m=truth.expected_settled_cube_center_world_m,
            trace=(),
        )

    def score_terminal(self, row, truth, trace):
        return evaluate_placement(
            row,
            execution_path=self.execution_path,
            truth=truth,
            settled_center_world_m=truth.expected_settled_cube_center_world_m
            + np.array([0.02, 0.0, 0.0]),
            trace=trace,
        )


class FlakyHealthGate:
    def __init__(self, fail_calls=0):
        self.fail_calls = fail_calls
        self.calls = []

    def probe(self, service_id, *, sam_image_path=None):
        self.calls.append((service_id, sam_image_path))
        if self.fail_calls:
            self.fail_calls -= 1
            raise ServiceGateError(service_id, "health", "connection refused")
        return {"service_id": service_id, "status": "PASS"}


class FakeIKController:
    grasp_offset_m = 0.024
    placement_offset_m = 0.027

    def __init__(self):
        self.requests = []

    def solve_ik(self, point, *, seed=None, maximum_iterations=250):
        self.requests.append((np.asarray(point).copy(), maximum_iterations))
        return np.asarray(seed, dtype=np.float64).copy()


class FakeRecorder:
    def __init__(
        self,
        _environment,
        _terminal,
        attempt_dir,
        *,
        callbacks,
        **_kwargs,
    ):
        self.attempt_dir = Path(attempt_dir)
        self.callbacks = callbacks
        self.running = False

    def start(self):
        self.running = True
        callbacks = self.callbacks
        if callbacks.evidence_event:
            callbacks.evidence_event({"kind": "FAKE_RECORDER_STARTED"})
        return self

    def stop(self):
        for name in ("initial.png", "final.png", "video.mp4"):
            atomic_write_bytes(
                self.attempt_dir / name, f"fake-{name}".encode(), overwrite=False
            )
        if self.callbacks.evidence_metadata:
            self.callbacks.evidence_metadata(
                {
                    "schema_version": "prefmem.video-metadata.v2",
                    "path": "video.mp4",
                    "initial_png": "initial.png",
                    "final_png": "final.png",
                    "width": 1920,
                    "height": 1080,
                    "fps": 10.0,
                }
            )
        self.running = False


class AuditableVideoRecorder(FakeRecorder):
    def stop(self):
        height, width = 1080, 1920
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (x + y) % 251
        frame[..., 1] = (2 * x + y) % 253
        frame[..., 2] = (x + 2 * y) % 255
        writer = cv2.VideoWriter(
            str(self.attempt_dir / "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("test MP4V writer is unavailable")
        writer.write(frame)
        writer.write(frame)
        writer.release()
        if not cv2.imwrite(str(self.attempt_dir / "initial.png"), frame):
            raise RuntimeError("could not write test initial PNG")
        if not cv2.imwrite(str(self.attempt_dir / "final.png"), frame):
            raise RuntimeError("could not write test final PNG")
        if self.callbacks.evidence_metadata:
            self.callbacks.evidence_metadata(
                {
                    "schema_version": "prefmem.video-metadata.v2",
                    "path": "video.mp4",
                    "initial_png": "initial.png",
                    "final_png": "final.png",
                    "width": width,
                    "height": height,
                    "fps": 10.0,
                    "frame_count": 2,
                    "duration_seconds": 0.2,
                    "codec": "mp4v",
                }
            )
        self.running = False


class PhysicalReadinessTests(unittest.TestCase):
    def test_canonical_separate_preflight_bundles_and_pf_log_validate(self):
        evidence = PhysicalPreflightEvidence.from_canonical_suite()
        self.assertNotEqual(
            evidence.reset_path.parent.resolve(),
            evidence.motion_path.parent.resolve(),
        )
        validated = evidence.validate(suite_root=SUITE_ROOT)
        self.assertEqual(
            set(validated),
            {"PF-RESET", "PF-MOTION", "PF-LOG", "frozen_configuration"},
        )
        self.assertEqual(validated["frozen_configuration"]["status"], "PASS")

    def test_physical_configuration_hash_pin_rejects_camera_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiments_suite_v2"
            _copy_physical_pins(root)
            self.assertEqual(
                validate_physical_configuration_pins(root)["status"], "PASS"
            )
            camera = (
                root
                / "camera_previews"
                / "motion_revision_candidate"
                / "camera_metadata.json"
            )
            camera.write_bytes(camera.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                PhysicalCampaignError,
                "camera_metadata.json",
            ):
                validate_physical_configuration_pins(root)

    def test_live_gate_accepts_exact_768_pair_without_control_change(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=True,
            viewer=False,
            realtime=True,
            width=768,
            height=768,
        )
        try:
            before = environment.snapshot().control.copy()
            report, sam, prefmem = verify_approved_live_task_streams(environment)
            after = environment.snapshot().control.copy()
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["approved_resolution"], [768, 768])
            self.assertTrue(report["synchronized_pair"])
            self.assertEqual(sam.rgb.shape, (768, 768, 3))
            self.assertEqual(prefmem.depth_m.shape, (768, 768))
            np.testing.assert_array_equal(after, before)
        finally:
            environment.close()

    def test_live_gate_rejects_nonapproved_resolution_without_motion(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=True,
            viewer=False,
            realtime=True,
            width=64,
            height=64,
        )
        try:
            before = environment.snapshot().control.copy()
            with self.assertRaisesRegex(
                PhysicalTaskStreamGateError,
                "approved 768 x 768",
            ):
                verify_approved_live_task_streams(environment)
            np.testing.assert_array_equal(
                environment.snapshot().control,
                before,
            )
        finally:
            environment.close()

    def test_dry_plan_is_exact_and_keeps_all_execution_side_effects_zero(self):
        plan = cal_x04_dry_plan()
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["run_or_attempt_allocated"])
        self.assertFalse(plan["simulator_started"])
        self.assertEqual(plan["motion_commands"], 0)
        self.assertEqual(plan["external_calls"], 0)
        self.assertEqual(plan["grid"]["planned_valid_terminal_rows"], 300)
        self.assertEqual(len(plan["grid"]["row_ids_in_execution_order"]), 300)
        self.assertEqual(plan["blockers"], ["CAL-X-03_NOT_PASSING"])

    def test_reset_precheck_runs_twenty_resets_and_rejects_hidden_corruption(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
        )
        try:
            report = run_pf_reset_precheck(environment)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["reset_count"], 20)
            self.assertTrue(
                report["checks"]["incorrect_hidden_arrangement_rejected"]
            )
            self.assertEqual(verify_tb6c_reset(environment)["status"], "PASS")
            environment.place_item("red_block", (0.35, -0.34, 0.026))
            with self.assertRaises(ResetVerificationError) as caught:
                verify_tb6c_reset(environment)
            self.assertFalse(caught.exception.report["checks"]["cube_positions_exact"])
        finally:
            environment.close()

    def test_motion_precheck_solves_every_exact_envelope_without_motion(self):
        controller = FakeIKController()
        report = run_pf_motion_precheck(controller)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["source_count"], 6)
        self.assertEqual(report["target_grid_point_count"], 10)
        self.assertEqual(report["ik_waypoint_count"], 32)
        self.assertEqual(len(controller.requests), 32)
        self.assertEqual(report["motion_commands"], 0)

    def test_materialized_scene_uses_relative_assets_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "experiments_suite_v2"
            scene = suite / "scenes" / "tb6c_v2.xml"
            scene.parent.mkdir(parents=True)
            shutil.copyfile(SCENE_PATH, scene)
            before = scene.read_bytes()
            destination = (
                suite
                / "results/shared-campaigns/r/CAL-X/A0/scene.xml"
            )
            materialize_v2_physical_scene(
                destination, scene_source=scene, suite_root=suite
            )
            payload = destination.read_text(encoding="utf-8")
            self.assertIn("assets/franka_emika_panda/mjx_panda.xml", payload)
            self.assertNotIn(str(suite), payload)
            self.assertEqual(scene.read_bytes(), before)

    def test_materialized_scene_compiles_from_real_portable_suite_tree(self):
        with tempfile.TemporaryDirectory(dir=SUITE_ROOT) as directory:
            destination = Path(directory) / "nested" / "attempt" / "scene.xml"
            materialize_v2_physical_scene(destination)
            model = __import__("mujoco").MjModel.from_xml_path(str(destination))
            self.assertGreater(model.nbody, 6)

    def test_full_profile_session_wires_reset_stage_perception_and_telemetry(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
            width=64,
            height=64,
            render_hz=20.0,
        )
        calibration = CameraCalibration(
            width=8,
            height=8,
            intrinsic=np.array(
                [[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]
            ),
            world_from_camera=np.eye(4),
        )
        frame = RGBDFrame(
            rgb=np.full((8, 8, 3), 64, dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32),
            calibration=calibration,
            observed_at=1.0,
            sequence=1,
            simulation_time=0.0,
        )
        environment.wait_for_sam_frame = lambda **_kwargs: frame
        environment.sam_rgbd_frame = lambda: frame
        try:
            with tempfile.TemporaryDirectory(dir=SUITE_ROOT) as directory:
                attempt = Path(directory) / "attempt"
                attempt.mkdir()
                row = build_cal_x04_rows()[0]
                session = FullPhysicalEvidenceSession(
                    attempt,
                    row=row,
                    environment=environment,
                    suite_run_id="test-run",
                    attempt_id="test-attempt.A0",
                    recorder_factory=FakeRecorder,
                    final_hold_seconds=0.0,
                )
                session.begin()
                truth = session.prepare_row()
                session.start_recorder()
                session.record_health((), status="NOT_APPLICABLE")
                evaluation = evaluate_placement(
                    row,
                    execution_path="ORACLE_CONTROLLER",
                    truth=truth,
                    settled_center_world_m=truth.expected_settled_cube_center_world_m,
                    trace=session.trace_records,
                )
                result = ResultEnvelope(
                    attempt_id="test-attempt.A0",
                    trial_tuple_sha256="0" * 64,
                    formal_verdict=FormalVerdict.PASS,
                    analytical_classification=AnalyticalClassification.PASS,
                    primary_oracle="test",
                    must_pass={"valid_terminal_measurement": True},
                    retry_lineage={
                        "retry_of": None,
                        "supersedes_attempt": None,
                    },
                )
                session.finish(result, evaluation)
                self.assertTrue((attempt / "initial_state.json").is_file())
                self.assertTrue((attempt / "final_state.json").is_file())
                self.assertTrue((attempt / "execution_trace.jsonl").is_file())
                self.assertTrue(
                    (attempt / "telemetry/state_snapshots.jsonl").is_file()
                )
                self.assertTrue((attempt / "perception/index.jsonl").is_file())
                self.assertTrue((attempt / "video_metadata.json").is_file())
                self.assertFalse((attempt / "model_calls.jsonl").exists())
        finally:
            environment.close()

    def test_controller_only_render_pause_spans_final_evidence_stop(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
            width=64,
            height=64,
            render_hz=20.0,
        )
        calibration = CameraCalibration(
            width=8,
            height=8,
            intrinsic=np.array(
                [[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]
            ),
            world_from_camera=np.eye(4),
        )
        frame = RGBDFrame(
            rgb=np.full((8, 8, 3), 64, dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32),
            calibration=calibration,
            observed_at=1.0,
            sequence=1,
            simulation_time=0.0,
        )
        events: list[tuple[str, object]] = []

        class AliveThread:
            def is_alive(self):
                return True

            def join(self, _timeout=None):
                return None

        class OrderedRecorder(FakeRecorder):
            def stop(recorder_self):
                events.append(
                    ("recorder_stop", environment.periodic_rendering_paused)
                )
                return super().stop()

        environment._thread = AliveThread()
        environment.wait_for_sam_frame = lambda **_kwargs: frame
        environment.sam_rgbd_frame = lambda: frame
        real_pause = environment.pause_periodic_rendering
        real_resume = environment.resume_periodic_rendering

        def pause_after_initial(**kwargs):
            persisted = list(
                iter_jsonl(attempt / "perception" / "index.jsonl")
            )
            events.append(
                (
                    "pause",
                    bool(persisted and persisted[-1]["endpoint"] == "INITIAL"),
                )
            )
            return real_pause(**kwargs)

        def exact_final_pair(**_kwargs):
            events.append(("final_pair", environment.periodic_rendering_paused))
            return frame, frame

        def resume_after_stop():
            events.append(("resume", environment.periodic_rendering_paused))
            return real_resume()

        environment.pause_periodic_rendering = pause_after_initial
        environment.render_camera_pair = exact_final_pair
        environment.resume_periodic_rendering = resume_after_stop
        try:
            with tempfile.TemporaryDirectory(dir=SUITE_ROOT) as directory:
                attempt = Path(directory) / "attempt"
                attempt.mkdir()
                row = build_cal_x04_rows()[0]
                session = FullPhysicalEvidenceSession(
                    attempt,
                    row=row,
                    environment=environment,
                    suite_run_id="pause-order-test",
                    attempt_id="pause-order-test.A0",
                    recorder_factory=OrderedRecorder,
                    final_hold_seconds=0.0,
                )
                session.begin()
                truth = session.prepare_row()
                session.start_recorder()
                evaluation = evaluate_placement(
                    row,
                    execution_path="ORACLE_CONTROLLER",
                    truth=truth,
                    settled_center_world_m=truth.expected_settled_cube_center_world_m,
                    trace=session.trace_records,
                )
                result = ResultEnvelope(
                    attempt_id="pause-order-test.A0",
                    trial_tuple_sha256="0" * 64,
                    formal_verdict=FormalVerdict.PASS,
                    analytical_classification=AnalyticalClassification.PASS,
                    primary_oracle="test",
                    must_pass={"valid_terminal_measurement": True},
                    retry_lineage={
                        "retry_of": None,
                        "supersedes_attempt": None,
                    },
                )
                session.finish(result, evaluation)
                self.assertEqual(
                    events,
                    [
                        ("pause", True),
                        ("final_pair", True),
                        ("recorder_stop", True),
                        ("resume", True),
                    ],
                )
                self.assertFalse(environment.periodic_rendering_paused)
        finally:
            environment._thread = None
            environment.close()

    def test_real_run_store_finalizes_and_audits_one_non_actuating_profile(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
            width=64,
            height=64,
        )
        calibration = CameraCalibration(
            width=8,
            height=8,
            intrinsic=np.array(
                [[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]
            ),
            world_from_camera=np.eye(4),
        )
        frame = RGBDFrame(
            rgb=np.full((8, 8, 3), 96, dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32),
            calibration=calibration,
            observed_at=1.0,
            sequence=1,
            simulation_time=0.0,
        )
        environment.wait_for_sam_frame = lambda **_kwargs: frame
        environment.sam_rgbd_frame = lambda: frame
        try:
            with tempfile.TemporaryDirectory(dir=SUITE_ROOT) as directory:
                store_root = Path(directory) / "store"
                _copy_physical_pins(store_root)
                preflight_root = store_root / "preflight"
                preflight = write_physical_preflight_bundle(
                    preflight_root,
                    reset_report={
                        "schema_version": "prefmem.pf-reset-precheck.v2",
                        "aim_id": "PF-RESET",
                        "status": "PASS",
                        "reset_count": 20,
                        "checks": {
                            "twenty_consecutive_resets_pass": True,
                            "incorrect_hidden_arrangement_rejected": True,
                            "frozen_state_restored_after_negative_control": True,
                        },
                        "agent_invocations": 0,
                        "motion_commands": 0,
                    },
                    motion_report={
                        "schema_version": "prefmem.pf-motion-precheck.v2",
                        "aim_id": "PF-MOTION",
                        "status": "PASS",
                        "source_count": 6,
                        "target_grid_point_count": 10,
                        "ik_waypoint_count": 32,
                        "controller_offsets_m": {
                            "grasp": 0.024,
                            "placement": 0.027,
                        },
                        "checks": {
                            "all_six_sources_covered": True,
                            "all_ten_target_grid_points_covered": True,
                            "all_waypoints_inside_workspace_fence": True,
                            "all_waypoints_ik_reachable": True,
                        },
                        "motion_commands": 0,
                    },
                    pf_log_report=_pf_log_report(),
                )
                pins = PhysicalCampaignPins(
                    suite_version="PrefMem-Experiment-Suite-v2",
                    source_commit="real-store-test",
                    reset_manifest_hash=EXPECTED_PHYSICAL_FIXTURE_SHA256,
                    scene_config_hash=EXPECTED_PHYSICAL_SCENE_SHA256,
                    robot_asset_id="robot",
                    model_config_hash="model",
                    prompt_tool_schema_hash="prompt",
                )
                store = RunStore(store_root)
                store.create_run(
                    "physical-audit-test",
                    source_commit=pins.source_commit,
                    dirty_state={"test": True},
                )

                def evidence_factory(
                    attempt_dir, *, row, suite_run_id, attempt_id
                ):
                    session = FullPhysicalEvidenceSession(
                        attempt_dir,
                        row=row,
                        environment=environment,
                        suite_run_id=suite_run_id,
                        attempt_id=attempt_id,
                        suite_root=SUITE_ROOT,
                        scene_source=SCENE_PATH,
                        recorder_factory=AuditableVideoRecorder,
                        final_hold_seconds=0.0,
                    )
                    # This test exercises profile finalization/audit, not the
                    # separately tested live 768 camera gate.
                    def verified_test_gate():
                        payload = {
                            "schema_version": (
                                "prefmem.cal-x04-task-stream-gate.v2"
                            ),
                            "status": "PASS",
                            "checked_before_motion": True,
                            "motion_dispatch_count_at_check": 0,
                            "approved_resolution": [768, 768],
                        }
                        atomic_write_json(
                            attempt_dir / "task_stream_pre_motion_gate.json",
                            payload,
                            overwrite=False,
                        )
                        return payload

                    session.verify_task_stream_gate = verified_test_gate
                    return session

                campaign = PhysicalCalibrationCampaign(
                    store=store,
                    suite_run_id="physical-audit-test",
                    aim_id="CAL-X-04",
                    pins=pins,
                    preflight=preflight,
                    evidence_factory=evidence_factory,
                    executor=FakeExecutor("ORACLE_CONTROLLER"),
                )
                outcome = campaign.run_next()
                self.assertTrue(outcome.audit_passed)
                self.assertTrue(
                    (outcome.attempt_dir / "artifact_manifest.json").is_file()
                )
                self.assertEqual(
                    load_json(outcome.attempt_dir / "attempt_manifest.json")[
                        "status"
                    ],
                    "FINALIZED",
                )
        finally:
            environment.close()


class PhysicalCampaignTests(unittest.TestCase):
    def test_high_rate_trace_writer_flushes_losslessly_within_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = root / "execution_trace.jsonl"
            telemetry = root / "telemetry" / "controller_trace.jsonl"
            writer = _BoundedHighRateTraceWriter(
                execution,
                telemetry,
                max_flush_interval_seconds=0.02,
            )
            records = [
                {
                    "schema_version": 1,
                    "kind": "JOINT_CONTROL_SAMPLE",
                    "sample_index": index,
                }
                for index in range(25)
            ]
            started = time.monotonic()
            try:
                for record in records:
                    writer.append(record)
                deadline = started + 0.5
                while writer.pending_count and time.monotonic() < deadline:
                    threading.Event().wait(0.005)
                self.assertEqual(writer.pending_count, 0)
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                writer.close()
            self.assertEqual(list(iter_jsonl(execution)), records)
            self.assertEqual(list(iter_jsonl(telemetry)), records)

    def test_authoritative_transition_drains_samples_in_strict_order(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
            width=64,
            height=64,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                attempt = Path(directory) / "attempt"
                attempt.mkdir()
                session = FullPhysicalEvidenceSession(
                    attempt,
                    row=build_cal_x04_rows()[0],
                    environment=environment,
                    suite_run_id="trace-order-test",
                    attempt_id="trace-order-test.A0",
                    recorder_factory=FakeRecorder,
                    final_hold_seconds=0.0,
                )
                for index in range(12):
                    session._on_execution_trace(
                        {
                            "schema_version": 1,
                            "kind": "JOINT_CONTROL_SAMPLE",
                            "sample_index": index,
                        }
                    )
                session._on_execution_trace(
                    {
                        "schema_version": 1,
                        "kind": "WAYPOINT_ACHIEVED",
                        "waypoint": "approach_source",
                    }
                )
                session._high_rate_writer.close()
                execution = list(iter_jsonl(attempt / "execution_trace.jsonl"))
                telemetry = list(
                    iter_jsonl(attempt / "telemetry" / "controller_trace.jsonl")
                )
                self.assertEqual(
                    [record["sample_index"] for record in execution[:-1]],
                    list(range(12)),
                )
                self.assertEqual(execution[-1]["kind"], "WAYPOINT_ACHIEVED")
                self.assertEqual(telemetry, execution)
        finally:
            environment.close()

    def test_full_physical_session_rejects_non_realtime_environment(self):
        environment = StackingEnvironment(
            scene_path=SCENE_PATH,
            start=False,
            viewer=False,
            realtime=False,
            width=64,
            height=64,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(
                    PhysicalCampaignError,
                    "realtime=True",
                ):
                    FullPhysicalEvidenceSession(
                        Path(directory) / "attempt",
                        row=build_cal_x04_rows()[0],
                        environment=environment,
                        suite_run_id="non-realtime-test",
                        attempt_id="non-realtime-test.A0",
                        recorder_factory=FakeRecorder,
                    )
        finally:
            environment.close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.suite = Path(self.temporary.name) / "experiments_suite_v2"
        self.suite.mkdir()
        _copy_physical_pins(self.suite)
        self.pins = PhysicalCampaignPins(
            suite_version="PrefMem-Experiment-Suite-v2",
            source_commit="abc123",
            reset_manifest_hash=EXPECTED_PHYSICAL_FIXTURE_SHA256,
            scene_config_hash=EXPECTED_PHYSICAL_SCENE_SHA256,
            robot_asset_id="robot-hash",
            model_config_hash="model-hash",
            prompt_tool_schema_hash="prompt-hash",
        )
        reset = self.suite / "preflight" / "pf_reset.json"
        self.preflight = write_physical_preflight_bundle(
            reset.parent,
            reset_report={
                "schema_version": "prefmem.pf-reset-precheck.v2",
                "aim_id": "PF-RESET",
                "status": "PASS",
                "reset_count": 20,
                "checks": {
                    "twenty_consecutive_resets_pass": True,
                    "incorrect_hidden_arrangement_rejected": True,
                    "frozen_state_restored_after_negative_control": True,
                },
                "agent_invocations": 0,
                "motion_commands": 0,
            },
            motion_report={
                "schema_version": "prefmem.pf-motion-precheck.v2",
                "aim_id": "PF-MOTION",
                "status": "PASS",
                "source_count": 6,
                "target_grid_point_count": 10,
                "ik_waypoint_count": 32,
                "controller_offsets_m": {
                    "grasp": 0.024,
                    "placement": 0.027,
                },
                "checks": {
                    "all_six_sources_covered": True,
                    "all_ten_target_grid_points_covered": True,
                    "all_waypoints_inside_workspace_fence": True,
                    "all_waypoints_ik_reachable": True,
                },
                "motion_commands": 0,
            },
            pf_log_report=_pf_log_report(),
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _audit(path):
        return AuditReport("ATTEMPT", str(path))

    def _campaign(self, aim_id, *, executor, health=None):
        store = FakeStore(self.suite, f"run-{aim_id}", self.pins)
        sessions = FakeSessionFactory()
        campaign = PhysicalCalibrationCampaign(
            store=store,
            suite_run_id=f"run-{aim_id}",
            aim_id=aim_id,
            pins=self.pins,
            preflight=self.preflight,
            evidence_factory=sessions,
            executor=executor,
            health_gate=health,
            audit=self._audit,
        )
        return campaign, store, sessions

    def test_exact_300_row_bundle_and_health_failure_becomes_linked_retry(self):
        health = FlakyHealthGate(fail_calls=1)
        campaign, _store, _sessions = self._campaign(
            "CAL-X-05",
            executor=FakeExecutor("PRODUCTION_FULL_PATH"),
            health=health,
        )
        inputs = list(iter_jsonl(campaign.campaign_dir / "input_rows.jsonl"))
        self.assertEqual(len(inputs), 300)
        self.assertEqual(
            [item["row"]["order_index"] for item in inputs], list(range(300))
        )

        invalid = campaign.run_next()
        self.assertEqual(invalid.attempt_number, 0)
        self.assertEqual(invalid.analytical_classification, "INVALID_RUN")
        retry = campaign.run_next()
        self.assertEqual(retry.row_id, invalid.row_id)
        self.assertEqual(retry.attempt_number, 1)
        self.assertEqual(retry.retry_of, invalid.attempt_id)
        first_manifest = load_json(invalid.attempt_dir / "attempt_manifest.json")
        retry_manifest = load_json(retry.attempt_dir / "attempt_manifest.json")
        self.assertEqual(
            first_manifest["trial_tuple"], retry_manifest["trial_tuple"]
        )
        second_row = campaign.run_next()
        self.assertNotEqual(second_row.row_id, retry.row_id)
        self.assertEqual(second_row.attempt_number, 0)
        self.assertTrue((invalid.attempt_dir / "artifact_manifest.json").is_file())

    def test_capability_failure_is_valid_and_never_retried(self):
        campaign, _store, _sessions = self._campaign(
            "CAL-X-04",
            executor=FakeExecutor("ORACLE_CONTROLLER", outcomes=("FAIL",)),
        )
        failed = campaign.run_next()
        self.assertEqual(failed.analytical_classification, "CAPABILITY_FAIL")
        next_outcome = campaign.run_next()
        self.assertNotEqual(next_outcome.row_id, failed.row_id)
        self.assertEqual(next_outcome.attempt_number, 0)
        aggregate = campaign.aggregate()
        self.assertEqual(aggregate["valid_terminal_measurements"], 2)
        self.assertEqual(
            aggregate["attempt_outcome_counts"],
            {"CAPABILITY_FAIL": 1, "PASS": 1},
        )

    def test_failed_live_gate_persists_required_artifact_before_invalid_finalize(self):
        store = FakeStore(self.suite, "camera-gate-fail", self.pins)
        executor = FakeExecutor("ORACLE_CONTROLLER")
        campaign = PhysicalCalibrationCampaign(
            store=store,
            suite_run_id="camera-gate-fail",
            aim_id="CAL-X-04",
            pins=self.pins,
            preflight=self.preflight,
            evidence_factory=FailingTaskStreamSessionFactory(),
            executor=executor,
            audit=self._audit,
        )
        outcome = campaign.run_next()
        self.assertEqual(outcome.analytical_classification, "INVALID_SETUP")
        gate = load_json(
            outcome.attempt_dir / "task_stream_pre_motion_gate.json"
        )
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(gate["checked_before_motion"])
        self.assertEqual(gate["motion_dispatch_count_at_check"], 0)
        self.assertEqual(executor.calls, [])
        result = load_json(outcome.attempt_dir / "results.json")
        self.assertEqual(
            result["invalidity"]["reason_code"],
            "TASK_STREAM_PRE_MOTION_GATE_FAILED",
        )

    def test_progress_snapshot_is_atomic_derived_and_reconstructible(self):
        campaign, _store, _sessions = self._campaign(
            "CAL-X-04", executor=FakeExecutor("ORACLE_CONTROLLER")
        )
        outcome = campaign.run_next()
        progress_path = campaign.campaign_dir / "progress.json"
        self.assertTrue(progress_path.is_file())
        progress = load_json(progress_path)
        self.assertEqual(progress["valid_terminal_measurements"], 1)
        self.assertEqual(progress["finalized_attempt_count"], 1)
        self.assertFalse(progress["attempt_artifact"])
        self.assertEqual(
            progress["source_attempts"][0]["attempt_id"],
            outcome.attempt_id,
        )
        self.assertEqual(
            progress["source_attempts"][0]["attempt_manifest_sha256"],
            sha256_file(outcome.attempt_dir / "attempt_manifest.json"),
        )
        self.assertEqual(
            progress["source_attempts"][0]["artifact_manifest_sha256"],
            sha256_file(outcome.attempt_dir / "artifact_manifest.json"),
        )
        self.assertEqual(progress, campaign.progress_snapshot())
        self.assertFalse((outcome.attempt_dir / "progress.json").exists())

    def test_unfinished_attempt_blocks_duplicate_allocation(self):
        campaign, store, _sessions = self._campaign(
            "CAL-X-04", executor=FakeExecutor("ORACLE_CONTROLLER")
        )
        attempt = store.allocate_attempt(
            campaign.suite_run_id,
            campaign.trials[0],
            campaign.case,
        )
        store.start_attempt(attempt)
        with self.assertRaises(PhysicalCampaignRecoveryRequired) as caught:
            campaign.run_next()
        self.assertEqual(caught.exception.attempt_dir, attempt)
        self.assertEqual(len(list((campaign.run_root / "CAL-X").iterdir())), 1)

    def test_fake_complete_campaign_aggregates_all_frozen_strata(self):
        campaign, _store, _sessions = self._campaign(
            "CAL-X-04", executor=FakeExecutor("ORACLE_CONTROLLER")
        )
        for _ in range(300):
            outcome = campaign.run_next()
            self.assertIsNotNone(outcome)
        self.assertIsNone(campaign.run_next())
        aggregate = campaign.aggregate()
        self.assertEqual(aggregate["valid_terminal_measurements"], 300)
        self.assertEqual(aggregate["gate"]["status"], "PASS")
        self.assertEqual(len(aggregate["gate"]["strata"]["cube_id"]), 6)
        self.assertEqual(len(aggregate["gate"]["strata"]["board_id"]), 2)
        self.assertEqual(len(aggregate["gate"]["strata"]["target_id"]), 5)
        path = campaign.write_final_aggregate()
        self.assertTrue(path.is_file())

    def test_paired_driver_holds_balanced_cursor_on_invalid_retry(self):
        store = FakeStore(self.suite, "paired-run", self.pins)
        controller = PhysicalCalibrationCampaign(
            store=store,
            suite_run_id="paired-run",
            aim_id="CAL-X-04",
            pins=self.pins,
            preflight=self.preflight,
            evidence_factory=FakeSessionFactory(),
            executor=FakeExecutor("ORACLE_CONTROLLER"),
            audit=self._audit,
            campaign_role="REGISTERED_PAIRED_CONTROL_REPLICATION",
        )
        health = FlakyHealthGate(fail_calls=1)
        full = PhysicalCalibrationCampaign(
            store=store,
            suite_run_id="paired-run",
            aim_id="CAL-X-05",
            pins=self.pins,
            preflight=self.preflight,
            evidence_factory=FakeSessionFactory(),
            executor=FakeExecutor("PRODUCTION_FULL_PATH"),
            health_gate=health,
            audit=self._audit,
            campaign_role="REGISTERED_PAIRED_FULL_PATH",
        )
        driver = BalancedPhysicalCampaignDriver(controller, full)
        with self.assertRaisesRegex(PhysicalCampaignError, "direct per-aim"):
            controller.run_next()
        with self.assertRaisesRegex(PhysicalCampaignError, "direct per-aim"):
            full.run_next()
        with self.assertRaisesRegex(PhysicalCampaignError, "balanced driver"):
            controller.write_final_aggregate()
        first = driver.run_next()
        self.assertEqual(first.analytical_classification, "PASS")
        self.assertEqual(first.row_id, controller.rows[0].row_id)
        invalid = driver.run_next()
        self.assertEqual(invalid.analytical_classification, "INVALID_RUN")
        self.assertEqual(invalid.row_id, full.rows[0].row_id)
        retry = driver.run_next()
        self.assertEqual(retry.row_id, invalid.row_id)
        self.assertEqual(retry.attempt_number, 1)
        next_entry = driver.status()
        self.assertEqual(next_entry["next_schedule_index"], 2)
        self.assertEqual(next_entry["next_aim_id"], "CAL-X-05")

    def test_phase_p_driver_uses_public_barrier_and_embargoes_interim_outcomes(self):
        class PublicCampaign:
            def __init__(self, aim_id, role, run_root, pins):
                self.aim_id = aim_id
                self.campaign_role = role
                self.run_root = run_root
                self.pins = pins
                self.rows = (
                    build_cal_x04_rows()
                    if aim_id == "CAL-X-04"
                    else build_cal_x05_rows()
                )
                self.completed = 0
                self.attempt_count = 0
                self.barrier_calls = 0
                self.gate_status = "INCOMPLETE"
                self.owner = None

            def bind_paired_execution(self, owner):
                self.owner = owner

            def status(self):
                return {
                    "aim_id": self.aim_id,
                    "planned_valid_rows": 300,
                    "valid_terminal_rows": self.completed,
                    "attempt_count": self.attempt_count,
                    "outcome_counts": {"PASS": self.completed},
                    "next_row_id": (
                        None
                        if self.completed == 300
                        else self.rows[self.completed].row_id
                    ),
                    "recovery_required": None,
                    "complete": self.completed == 300,
                }

            def run_next_paired(self, owner):
                if owner is not self.owner:
                    raise AssertionError("wrong paired owner")
                if self.aim_id == "CAL-X-05":
                    self.barrier_calls += 1
                row = self.rows[self.completed]
                self.completed += 1
                self.attempt_count += 1
                return SimpleNamespace(row_id=row.row_id, valid=True)

            def aggregate(self):
                return {
                    "aim_id": self.aim_id,
                    "valid_terminal_measurements": self.completed,
                    "unresolved_row_ids": [
                        row.row_id for row in self.rows[self.completed :]
                    ],
                    "gate": {"status": self.gate_status},
                }

        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "paired-run"
            pins = object()
            controller = PublicCampaign(
                "CAL-X-04",
                "REGISTERED_PAIRED_CONTROL_REPLICATION",
                run_root,
                pins,
            )
            full = PublicCampaign(
                "CAL-X-05", "REGISTERED_PAIRED_FULL_PATH", run_root, pins
            )
            driver = BalancedPhysicalCampaignDriver(controller, full)
            outcomes = [driver.run_next() for _ in range(4)]
            self.assertEqual(
                [outcome.row_id.split("-")[2] for outcome in outcomes],
                ["04", "05", "05", "04"],
            )
            self.assertEqual(full.barrier_calls, 2)
            interim = driver.aggregate()
            self.assertEqual(interim["analysis_embargo"], "ACTIVE")
            self.assertEqual(
                interim["CAL-X-05"]["status"],
                "WITHHELD_PENDING_600_VALID_TERMINALS",
            )
            self.assertNotIn("gate", interim["CAL-X-05"])

            controller.completed = full.completed = 300
            controller.gate_status = "FAIL"
            full.gate_status = "PASS"
            terminal = driver.aggregate()
            self.assertTrue(terminal["complete"])
            self.assertEqual(terminal["analysis_embargo"], "RELEASED")
            self.assertEqual(terminal["downstream_qualification"], "QUARANTINED")
            self.assertTrue(
                terminal["cal_x05_observations_immutable_but_quarantined"]
            )

            mismatched = PublicCampaign(
                "CAL-X-05", "REGISTERED_PAIRED_FULL_PATH", run_root, object()
            )
            with self.assertRaisesRegex(ValueError, "share frozen pins"):
                BalancedPhysicalCampaignDriver(controller, mismatched)


if __name__ == "__main__":
    unittest.main()
