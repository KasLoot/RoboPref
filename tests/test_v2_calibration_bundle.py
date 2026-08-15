from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.runners import calibration_bundle as bundle_module
from experiments_suite_v2.runners.bundles import verify_row_bundle
from experiments_suite_v2.runners.calibration import (
    CalibrationEvaluation,
    ObjectCalibrationObservation,
    build_cal_x01_rows,
    build_cal_x02_rows,
    build_cal_x03_rows,
)
from experiments_suite_v2.runners.calibration_bundle import (
    CalibrationSamInterruption,
    aggregate_cal_x03_full_placement_gate,
    run_cal_x01_bundle,
    run_cal_x02_bundle,
    run_paired_calibration_bundle,
    run_paired_calibration_retry,
)
from prefmem.execution.frames import CameraCalibration, RGBDFrame
from prefmem.execution.sam import SamDetection, SamServiceError


def _mask() -> np.ndarray:
    value = np.zeros((768, 768), dtype=bool)
    value[352:416, 352:416] = True
    return value


def _observation(row) -> ObjectCalibrationObservation:
    mask = _mask()
    calibration = CameraCalibration(
        width=768,
        height=768,
        intrinsic=np.array(
            [[800.0, 0.0, 383.5], [0.0, 800.0, 383.5], [0.0, 0.0, 1.0]]
        ),
        world_from_camera=np.eye(4),
    )
    frame = RGBDFrame(
        rgb=np.zeros((768, 768, 3), dtype=np.uint8),
        depth_m=np.ones((768, 768), dtype=np.float32),
        calibration=calibration,
        observed_at=float(row.order_index + 1),
        sequence=row.order_index + 1,
        simulation_time=0.0,
    )
    return ObjectCalibrationObservation(
        shared_frame_id=row.shared_frame_id,
        object_id=row.object_id,
        selector=row.selector,
        anchor=row.anchor,
        geom_id=7,
        frame=frame,
        oracle_mask=mask,
        hidden_true_anchor_world_m=np.array([0.0, 0.0, -1.0]),
    )


class FixtureSam:
    def __init__(self, *, failures: int = 0) -> None:
        self.threshold = 0.5
        self.failures = failures
        self.calls = 0

    def detect(self, rgb, prompt, *, threshold=None):
        del rgb, prompt, threshold
        self.calls += 1
        if self.calls <= self.failures:
            raise SamServiceError("fixture tunnel disconnected")
        mask = _mask()
        return (
            SamDetection(
                object_id=7,
                box_xyxy=(352, 352, 416, 416),
                mask_area=int(mask.sum()),
                mask=mask,
                score=0.99,
            ),
        )


class EmptySam:
    threshold = 0.5

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, rgb, prompt, *, threshold=None):
        del rgb, prompt, threshold
        self.calls += 1
        return ()


class CalibrationBundleTests(unittest.TestCase):
    def _rows(self):
        return (build_cal_x02_rows()[0],), (build_cal_x03_rows()[0],)

    def test_paired_bundle_allocates_first_and_hashes_exact_same_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "CAL-X-02-X03__A0"
            rows_x02, rows_x03 = self._rows()
            captures = 0

            def capture(row):
                nonlocal captures
                captures += 1
                self.assertTrue((output / "attempt_manifest.json").is_file())
                self.assertTrue((output / "inputs.json").is_file())
                return _observation(row)

            summary = run_paired_calibration_bundle(
                FixtureSam(),
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=capture,
                development_subset=True,
            )
            self.assertEqual(captures, 1)
            self.assertTrue(summary["same_frame_pairing_verified"])
            self.assertEqual(
                summary["cal_x03"]["full_placement_gate"]["limits"],
                {
                    "absolute_median_signed_x_y_mm": 5.0,
                    "p95_radial_xy_mm": 12.0,
                },
            )
            self.assertEqual(
                summary["cal_x03"]["full_placement_gate"]["gate_status"],
                "INCOMPLETE_GRID",
            )
            left = load_json(next((output / "rows" / "cal_x_02").glob("*.json")))
            right = load_json(next((output / "rows" / "cal_x_03").glob("*.json")))
            self.assertEqual(
                left["paired_frame_evidence_sha256"],
                right["paired_frame_evidence_sha256"],
            )
            inputs = load_json(output / "inputs.json")
            self.assertEqual(inputs["camera"]["resolution"], [768, 768])
            for identity in (
                inputs["scene"]["sha256"],
                inputs["camera"]["runtime_freeze_sha256"],
                inputs["camera"]["approved_configuration_sha256"],
                inputs["services"]["registry_sha256"],
                inputs["services"]["sam_service_identity_sha256"],
            ):
                self.assertRegex(identity, r"^[0-9a-f]{64}$")
            self.assertEqual(verify_row_bundle(output), [])

    def test_service_health_gate_runs_after_allocation_and_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "A1"
            rows_x02, rows_x03 = self._rows()
            order: list[str] = []

            def health():
                self.assertTrue((output / "attempt_manifest.json").is_file())
                order.append("health")
                return {
                    "schema_version": "prefmem.service-health.v2",
                    "service_id": "sam3.1-grounding-v1",
                    "kind": "SAM_SEGMENTATION",
                    "status": "PASS",
                    "observed_at": "fixture",
                    "evidence": {
                        "health": {
                            "status": "ok",
                            "model": "sam3.1",
                            "checkpoint": (
                                "/workspace/models/sam3.1/"
                                "sam3.1_multiplex.pt"
                            ),
                        }
                    },
                }

            def capture(row):
                order.append("capture")
                return _observation(row)

            summary = run_paired_calibration_bundle(
                FixtureSam(),
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=capture,
                development_subset=True,
                pre_sam_health_gate=health,
                scene_revision_id="TB6C-v2-PF-MOTION-01",
                prerequisite_evidence={"PF-SELECTOR": "PASS"},
            )
            self.assertEqual(order, ["health", "capture"])
            self.assertTrue((output / "service_health.json").is_file())
            inputs = load_json(output / "inputs.json")
            self.assertEqual(
                inputs["scene_revision_id"], "TB6C-v2-PF-MOTION-01"
            )
            self.assertEqual(
                inputs["prerequisite_evidence"], {"PF-SELECTOR": "PASS"}
            )

            # A finalized call is verification-only and cannot render/call again.
            cached = run_paired_calibration_bundle(
                FixtureSam(failures=10),
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=lambda _row: self.fail("finalized bundle recaptured a frame"),
                development_subset=True,
                pre_sam_health_gate=health,
                scene_revision_id="TB6C-v2-PF-MOTION-01",
                prerequisite_evidence={"PF-SELECTOR": "PASS"},
            )
            self.assertEqual(cached, summary)

            # A crash between the artifact manifest and suite checksum is
            # recoverable without changing any immutable payload artifact.
            artifact_manifest_hash = sha256_file(output / "artifact_manifest.json")
            (output / "checksums.sha256").unlink()
            recovered = run_paired_calibration_bundle(
                FixtureSam(failures=10),
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=lambda _row: self.fail("finalization recovery rendered"),
                development_subset=True,
                pre_sam_health_gate=health,
                scene_revision_id="TB6C-v2-PF-MOTION-01",
                prerequisite_evidence={"PF-SELECTOR": "PASS"},
            )
            self.assertEqual(recovered, summary)
            self.assertEqual(
                sha256_file(output / "artifact_manifest.json"),
                artifact_manifest_hash,
            )
            self.assertEqual(verify_row_bundle(output), [])

    def test_sam_outage_resumes_same_attempt_from_committed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "CAL-X-02-X03__A0"
            rows_x02, rows_x03 = self._rows()
            detector = FixtureSam(failures=1)
            captures = 0

            def capture(row):
                nonlocal captures
                captures += 1
                return _observation(row)

            with self.assertRaises(CalibrationSamInterruption) as raised:
                run_paired_calibration_bundle(
                    detector,
                    output,
                    rows_x02=rows_x02,
                    rows_x03=rows_x03,
                    capture=capture,
                    development_subset=True,
                )
            self.assertTrue(raised.exception.safe_same_attempt_resume)
            manifest = load_json(output / "attempt_manifest.json")
            self.assertEqual(manifest["status"], "INVALID_RUN")
            self.assertEqual(manifest["reason_code"], "SAM_CONNECTION_LOST")
            evidence = output / "perception" / rows_x02[0].shared_frame_id / "frame_evidence.npz"
            before_evidence = sha256_file(evidence)
            oracle_row = next((output / "rows" / "cal_x_02").glob("*.json"))
            before_oracle = sha256_file(oracle_row)

            summary = run_paired_calibration_bundle(
                detector,
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=capture,
                development_subset=True,
            )
            self.assertEqual(captures, 1)
            self.assertEqual(detector.calls, 2)
            self.assertEqual(sha256_file(evidence), before_evidence)
            self.assertEqual(sha256_file(oracle_row), before_oracle)
            self.assertTrue(summary["same_frame_pairing_verified"])
            final_manifest = load_json(output / "attempt_manifest.json")
            self.assertTrue(final_manifest["resumed_from_invalid_run"])
            self.assertEqual(verify_row_bundle(output), [])

    def test_usable_zero_detection_is_durable_capability_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "A1"
            rows_x02, rows_x03 = self._rows()
            detector = EmptySam()
            summary = run_paired_calibration_bundle(
                detector,
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=_observation,
                development_subset=True,
            )
            self.assertEqual(detector.calls, 1)
            incremental = summary["cal_x03"]["incremental_metrics"]
            self.assertEqual(incremental["sam_grounding_success_count"], 0)
            self.assertEqual(incremental["sam_grounding_failure_count"], 1)
            gate = summary["cal_x03"]["full_placement_gate"]
            self.assertFalse(gate["all_rows_grounded"])
            self.assertEqual(gate["threshold_status"], "FAIL")
            result = load_json(output / "results.json")
            self.assertEqual(result["analytical_verdict"], "CAPABILITY_FAIL")
            self.assertEqual(verify_row_bundle(output), [])

    def test_committed_sam_response_is_replayed_after_row_checkpoint_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "CAL-X-02-X03__A0"
            rows_x02, rows_x03 = self._rows()
            detector = FixtureSam()
            original_checkpoint = bundle_module._write_row_checkpoint

            def crash_before_x03_checkpoint(output_dir, row):
                if row["aim_id"] == "CAL-X-03":
                    raise RuntimeError("fixture crash after SAM response commit")
                return original_checkpoint(output_dir, row)

            with mock.patch.object(
                bundle_module, "_write_row_checkpoint",
                side_effect=crash_before_x03_checkpoint,
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture crash"):
                    run_paired_calibration_bundle(
                        detector,
                        output,
                        rows_x02=rows_x02,
                        rows_x03=rows_x03,
                        capture=_observation,
                        development_subset=True,
                    )
            self.assertEqual(detector.calls, 1)
            response = output / "perception" / rows_x02[0].shared_frame_id / "sam_response.json"
            self.assertTrue(response.is_file())
            response_hash = sha256_file(response)

            no_live_call = FixtureSam(failures=100)
            summary = run_paired_calibration_bundle(
                no_live_call,
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=lambda _row: self.fail("resume recaptured frame"),
                development_subset=True,
            )
            self.assertEqual(no_live_call.calls, 0)
            self.assertEqual(sha256_file(response), response_hash)
            self.assertTrue(summary["same_frame_pairing_verified"])

    def test_row_commit_survives_crash_before_manifest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "CAL-X-02-X03__A0"
            rows_x02, rows_x03 = self._rows()
            detector = FixtureSam()
            original_checkpoint = bundle_module._write_row_checkpoint

            def commit_then_crash(output_dir, row):
                original_checkpoint(output_dir, row)
                if row["aim_id"] == "CAL-X-03":
                    raise RuntimeError("fixture crash before manifest checkpoint")

            with mock.patch.object(
                bundle_module, "_write_row_checkpoint",
                side_effect=commit_then_crash,
            ):
                with self.assertRaisesRegex(RuntimeError, "manifest checkpoint"):
                    run_paired_calibration_bundle(
                        detector,
                        output,
                        rows_x02=rows_x02,
                        rows_x03=rows_x03,
                        capture=_observation,
                        development_subset=True,
                    )
            self.assertEqual(load_json(output / "attempt_manifest.json")["completed_rows"], 0)
            self.assertTrue(next((output / "rows" / "cal_x_03").glob("*.json")).is_file())

            no_live_call = FixtureSam(failures=100)
            run_paired_calibration_bundle(
                no_live_call,
                output,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=lambda _row: self.fail("row recovery recaptured frame"),
                development_subset=True,
            )
            self.assertEqual(no_live_call.calls, 0)
            self.assertEqual(load_json(output / "attempt_manifest.json")["completed_rows"], 1)
            self.assertEqual(verify_row_bundle(output), [])

    def test_unknown_inflight_sam_outcome_requires_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "CAL-X-02-X03__A0"
            rows_x02, rows_x03 = self._rows()
            detector = FixtureSam()
            original_write = bundle_module.atomic_write_json

            def crash_before_response_commit(path, value, *, overwrite=True):
                if Path(path).name == "sam_response.json":
                    raise RuntimeError("fixture crash before SAM response commit")
                return original_write(path, value, overwrite=overwrite)

            with mock.patch.object(
                bundle_module,
                "atomic_write_json",
                side_effect=crash_before_response_commit,
            ):
                with self.assertRaisesRegex(RuntimeError, "response commit"):
                    run_paired_calibration_bundle(
                        detector,
                        output,
                        rows_x02=rows_x02,
                        rows_x03=rows_x03,
                        capture=_observation,
                        development_subset=True,
                    )
            self.assertEqual(detector.calls, 1)
            self.assertTrue(
                (
                    output
                    / "perception"
                    / rows_x02[0].shared_frame_id
                    / "sam_request_000.json"
                ).is_file()
            )
            no_resample = FixtureSam(failures=100)
            with self.assertRaises(CalibrationSamInterruption) as raised:
                run_paired_calibration_bundle(
                    no_resample,
                    output,
                    rows_x02=rows_x02,
                    rows_x03=rows_x03,
                    capture=lambda _row: self.fail("unknown outcome recaptured frame"),
                    development_subset=True,
                )
            self.assertFalse(raised.exception.safe_same_attempt_resume)
            self.assertEqual(raised.exception.reason_code, "SAM_CALL_OUTCOME_UNKNOWN")
            self.assertEqual(no_resample.calls, 0)
            manifest = load_json(output / "attempt_manifest.json")
            self.assertEqual(manifest["status"], "INVALID_RUN")
            self.assertFalse(manifest["safe_same_attempt_resume"])

    def test_strict_invalid_retry_creates_exact_a1_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "CAL-X-02-X03__A0"
            retry = root / "CAL-X-02-X03__A1"
            rows_x02, rows_x03 = self._rows()
            detector = FixtureSam(failures=1)
            with self.assertRaises(CalibrationSamInterruption):
                run_paired_calibration_bundle(
                    detector,
                    original,
                    rows_x02=rows_x02,
                    rows_x03=rows_x03,
                    capture=_observation,
                    development_subset=True,
                )
            run_paired_calibration_retry(
                detector,
                original,
                retry,
                rows_x02=rows_x02,
                rows_x03=rows_x03,
                capture=_observation,
                development_subset=True,
            )
            a0 = load_json(original / "attempt_manifest.json")
            a1 = load_json(retry / "attempt_manifest.json")
            self.assertEqual(
                load_json(original / "results.json")["analytical_verdict"],
                "INVALID_RUN",
            )
            self.assertEqual(a0["attempt_id"], "CAL-X-02-X03__A0")
            self.assertEqual(a1["attempt_id"], "CAL-X-02-X03__A1")
            self.assertEqual(a1["retry_of"], a0["attempt_id"])
            self.assertEqual(a1["supersedes_attempt"], a0["attempt_id"])
            self.assertEqual(a1["frozen"], a0["frozen"])
            self.assertEqual(verify_row_bundle(original), [])
            self.assertEqual(verify_row_bundle(retry), [])

    def test_cal_x01_and_x02_helpers_are_allocation_first_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x01_output = root / "CAL-X-01__A0"
            x01_row = build_cal_x01_rows()[0]

            def evaluate(row):
                self.assertTrue((x01_output / "attempt_manifest.json").is_file())
                return CalibrationEvaluation(
                    "CAL-X-01",
                    row,
                    telemetry={"fixture": True},
                    scoring={
                        "rendered_roundtrip_error_mm": [0.0, 0.0, 0.0],
                        "axis_sign_pass": True,
                    },
                )

            x01 = run_cal_x01_bundle(
                x01_output,
                rows=(x01_row,),
                evaluate=evaluate,
                development_subset=True,
                attempt_id="A1",
                scene_revision_id="TB6C-v2-PF-MOTION-01",
            )
            self.assertEqual(x01["cal_x01"]["status"], "PASS")
            self.assertEqual(x01["attempt_id"], "A1")
            self.assertEqual(
                load_json(x01_output / "inputs.json")["scene_revision_id"],
                "TB6C-v2-PF-MOTION-01",
            )
            self.assertEqual(verify_row_bundle(x01_output), [])

            x02_output = root / "CAL-X-02__A0"
            x02_row = build_cal_x02_rows()[0]
            x02 = run_cal_x02_bundle(
                x02_output,
                rows=(x02_row,),
                capture=_observation,
                development_subset=True,
            )
            self.assertEqual(x02["cal_x02"]["status"], "PASS")
            self.assertEqual(verify_row_bundle(x02_output), [])

    def test_full_placement_gate_uses_inclusive_five_and_twelve_mm_limits(self) -> None:
        passing = [
            {
                "scoring": {
                    "sam_anchor_error_mm": [5.0, -5.0, 100.0],
                    "sam_radial_xy_error_mm": 12.0,
                }
            }
            for _ in range(20)
        ]
        gate = aggregate_cal_x03_full_placement_gate(
            passing, complete_registered_grid=True
        )
        self.assertEqual(gate["threshold_status"], "PASS")
        self.assertEqual(gate["gate_status"], "PASS")
        failing = [
            {
                "scoring": {
                    "sam_anchor_error_mm": [5.01, 0.0, 0.0],
                    "sam_radial_xy_error_mm": 12.01,
                }
            }
            for _ in range(20)
        ]
        self.assertEqual(
            aggregate_cal_x03_full_placement_gate(
                failing, complete_registered_grid=True
            )["threshold_status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
