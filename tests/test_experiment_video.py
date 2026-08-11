from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from experiments.harness.recording import AttemptRecorder, ClockOrigin
from experiments.harness.campaign import RunStatus as CampaignRunStatus
from experiments.harness.video import (
    OverlayState,
    SourceFrameReference,
    SynchronizedVideoRecorder,
    VideoConfig,
    VideoError,
    audit_attempt_videos,
    audit_video_file,
    compose_evidence_frame,
    source_frame_sha256,
)


def _third(step: int) -> np.ndarray:
    frame = np.full((360, 640, 3), (30, 45, 65), dtype=np.uint8)
    cv2.rectangle(
        frame, (40 + step * 12, 100), (180 + step * 12, 240), (20, 180, 240), -1
    )
    cv2.putText(
        frame,
        f"scene-{step}",
        (230, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return frame


def _robot(step: int) -> np.ndarray:
    frame = np.full((240, 320, 3), (70, 35, 20), dtype=np.uint8)
    cv2.circle(frame, (80 + step * 15, 120), 38, (50, 230, 90), -1)
    return frame


def _png(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise AssertionError("test fixture PNG did not encode")
    return encoded.tobytes()


def _jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not ok:
        raise AssertionError("test fixture JPEG did not encode")
    return encoded.tobytes()


def _overlay(step: int, *, secret: str = "") -> OverlayState:
    return OverlayState(
        episode="S0001/attempt_1",
        profile="T5_full",
        scenario="nominal_pick_place",
        seed=17,
        system_state="EXECUTING" if step else "RESETTING",
        phase="STEP",
        active_agent="Monitor",
        latest_user_request="put the red block on the mat",
        latest_system_output="executing step",
        safety_event="none",
        monitor_event="clear",
        validator_event="pending",
        terminal_lines=("E000001 scene reset", f"Authorization: Bearer {secret}"),
    )


class ExperimentVideoTests(unittest.TestCase):
    def test_source_jitter_budget_is_hard_capped_at_one_tenth_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "one tenth"):
            VideoConfig(source_timestamp_jitter_seconds=0.010001)

    def test_composite_uses_dedicated_unobscured_scene_panes(self) -> None:
        frame = compose_evidence_frame(
            _third(0),
            _robot(0),
            overlay=_overlay(0),
            utc="2026-08-11T00:00:00.000Z",
            elapsed_seconds=0.0,
        )
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(float(frame[40:720, :1280].mean()), 1)
        self.assertGreater(float(frame[40:520, 1280:].mean()), 1)
        self.assertGreater(float(frame[720:, :].std()), 1)

    def test_synchronized_h264_video_and_ffprobe_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = ClockOrigin.capture()
            recorder = SynchronizedVideoRecorder(
                root,
                origin=origin,
                config=VideoConfig(pre_roll_seconds=0.2, final_hold_seconds=0.2),
                secret_values=("video-secret",),
            )
            for step, elapsed in enumerate((0.4, 0.5, 0.6)):
                recorder.write(
                    _third(step),
                    _robot(step),
                    overlay=_overlay(step, secret="video-secret"),
                    source_elapsed_seconds=elapsed,
                    source_reference=SourceFrameReference(
                        third_person_id=f"third-{step}",
                        robot_camera_id=f"robot-{step}",
                    ),
                )
            metadata = recorder.finalize()
            self.assertEqual(metadata["composite"]["frame_count"], 7)
            self.assertEqual(metadata["robot_camera"]["frame_count"], 7)
            self.assertEqual(len(metadata["source_frame_linkage"]), 7)
            self.assertGreater((root / "video.mp4").stat().st_size, 0)
            self.assertGreater((root / "robot_camera.mp4").stat().st_size, 0)
            self.assertFalse((root / "video.mp4.partial").exists())
            self.assertFalse((root / "robot_camera.mp4.partial").exists())

            composite = audit_video_file(
                root / "video.mp4",
                expected_width=1920,
                expected_height=1080,
                expected_fps=10,
                expected_frames=7,
            )
            self.assertTrue(composite["passed"], composite["errors"])
            audit = audit_attempt_videos(root)
            self.assertTrue(audit["passed"], audit["errors"])
            self.assertEqual(audit["composite"]["codec"], "h264")
            self.assertTrue(all(audit["composite"]["decodable"].values()))
            self.assertNotIn(b"video-secret", (root / "video_metadata.json").read_bytes())

    def test_audit_rejects_linkage_enum_and_pre_roll_relabel_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedVideoRecorder(
                root,
                origin=ClockOrigin.capture(),
                config=VideoConfig(pre_roll_seconds=0.2, final_hold_seconds=0),
            )
            for step, elapsed in enumerate((0.4, 0.5)):
                recorder.write(
                    _third(step),
                    _robot(step),
                    overlay=_overlay(step),
                    source_elapsed_seconds=elapsed,
                    source_reference=SourceFrameReference(
                        third_person_id=f"third-{step}",
                        robot_camera_id=f"robot-{step}",
                    ),
                )
            original = recorder.finalize()
            self.assertTrue(audit_attempt_videos(root)["passed"])
            metadata_path = root / "video_metadata.json"

            def assert_tamper(mutator, expected_error: str) -> None:
                tampered = deepcopy(original)
                mutator(tampered["source_frame_linkage"])
                metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
                audit = audit_attempt_videos(root)
                self.assertFalse(audit["passed"])
                self.assertTrue(
                    any(expected_error in error for error in audit["errors"]),
                    audit["errors"],
                )

            assert_tamper(
                lambda links: links[2].__setitem__("source_kind", ["camera"]),
                "invalid source_kind",
            )
            assert_tamper(
                lambda links: links[0].__setitem__(
                    "hold_kind", {"kind": "source_gap"}
                ),
                "invalid hold_kind",
            )
            assert_tamper(
                lambda links: links[-1].__setitem__("hold_kind", "pre_roll"),
                "contiguous leading prefix",
            )
            assert_tamper(
                lambda links: links[0].__setitem__(
                    "robot_camera_id", "forged-pre-roll"
                ),
                "pre_roll frame changes the first source identity",
            )
            assert_tamper(
                lambda links: links[0].__setitem__(
                    "source_capture_elapsed_seconds", 0.3
                ),
                "pre_roll frame changes the first source acquisition time",
            )

    def test_declared_source_hash_must_match_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(pre_roll_seconds=0),
            )
            with self.assertRaisesRegex(VideoError, "source hash is not exact"):
                recorder.write(
                    _third(0),
                    _robot(0),
                    overlay=_overlay(0),
                    source_elapsed_seconds=0.1,
                    source_reference=SourceFrameReference(
                        third_person_sha256="0" * 64
                    ),
                )
            marker = recorder.abort(reason="controlled hash mismatch")
            self.assertTrue(marker.is_file())
            self.assertEqual(json.loads(marker.read_text())["artifact_state"], "PARTIAL")

    def test_bounded_external_cfr_jitter_snaps_without_losing_capture_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0,
                    source_timestamp_jitter_seconds=0.01,
                ),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=0.0,
            )
            recorder.write(
                _third(1),
                _robot(1),
                overlay=_overlay(1),
                source_elapsed_seconds=0.095,
            )
            metadata = recorder.finalize()
            second = metadata["source_frame_linkage"][1]
            self.assertAlmostEqual(second["source_elapsed_seconds"], 0.1)
            self.assertAlmostEqual(
                second["source_capture_elapsed_seconds"], 0.095
            )

    def test_early_jitter_collision_fails_closed_at_writer_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0,
                    source_timestamp_jitter_seconds=0.01,
                ),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=0.051,
            )
            with self.assertRaisesRegex(VideoError, "increasing CFR slot"):
                recorder.write(
                    _third(1),
                    _robot(1),
                    overlay=_overlay(1),
                    source_elapsed_seconds=0.141,
                )
            recorder.abort(reason="controlled same-slot producer frame")

    def test_periodic_external_source_at_arbitrary_cfr_phase_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            for step, elapsed in enumerate((2.379, 2.479, 2.579)):
                recorder.write(
                    _third(step),
                    _robot(step),
                    overlay=_overlay(step),
                    source_elapsed_seconds=elapsed,
                )
            metadata = recorder.finalize()
            acquisitions = [
                item
                for item in metadata["source_frame_linkage"]
                if item["hold_kind"] is None
            ]
            self.assertEqual(
                [item["source_elapsed_seconds"] for item in acquisitions],
                [2.4, 2.5, 2.6],
            )
            self.assertEqual(
                [item["source_capture_elapsed_seconds"] for item in acquisitions],
                [2.379, 2.479, 2.579],
            )

    def test_overlay_update_retains_source_acquisition_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=0.0,
                source_reference=SourceFrameReference(
                    third_person_id="third-source",
                    robot_camera_id="robot-source",
                ),
            )
            recorder.advance_overlay(overlay=_overlay(1))
            metadata = recorder.finalize()
            overlay_link = next(
                item
                for item in metadata["source_frame_linkage"]
                if item["hold_kind"] == "overlay_update"
            )
            self.assertEqual(overlay_link["source_capture_elapsed_seconds"], 0.0)
            self.assertEqual(overlay_link["third_person_id"], "third-source")
            self.assertEqual(overlay_link["robot_camera_id"], "robot-source")
            audit = audit_attempt_videos(Path(directory))
            self.assertTrue(audit["passed"], audit["errors"])

    def test_external_source_rate_beyond_jitter_budget_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedVideoRecorder(
                Path(directory),
                origin=ClockOrigin.capture(),
                config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0,
                    source_timestamp_jitter_seconds=0.01,
                ),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=0.0,
            )
            with self.assertRaisesRegex(VideoError, "exceed"):
                recorder.write(
                    _third(1),
                    _robot(1),
                    overlay=_overlay(1),
                    source_elapsed_seconds=0.05,
                )
            recorder.abort(reason="controlled over-rate source")

    def test_audit_counts_source_gap_holds_as_frozen_camera_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedVideoRecorder(
                root,
                origin=ClockOrigin.capture(),
                config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0,
                    maximum_declared_freeze_seconds=0.2,
                ),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=0.0,
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(1),
                source_elapsed_seconds=0.6,
            )
            recorder.finalize()
            audit = audit_attempt_videos(root)
            self.assertFalse(audit["passed"])
            self.assertTrue(
                any("frozen source panes" in item for item in audit["errors"]),
                audit["errors"],
            )

    def test_audit_accepts_static_pixels_from_distinct_fresh_acquisitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedVideoRecorder(
                root,
                origin=ClockOrigin.capture(),
                config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0,
                    maximum_declared_freeze_seconds=0.2,
                ),
            )
            for sequence in range(6):
                recorder.write(
                    _third(0),
                    _robot(0),
                    overlay=_overlay(0),
                    source_elapsed_seconds=sequence / 10,
                    source_reference=SourceFrameReference(
                        third_person_id=f"overview:{sequence}",
                        robot_camera_id=f"robot_camera:{sequence}",
                    ),
                )
            recorder.finalize()
            audit = audit_attempt_videos(root)
            self.assertFalse(
                any("frozen source panes" in item for item in audit["errors"]),
                audit["errors"],
            )
            self.assertTrue(audit["passed"], audit["errors"])

    def test_audit_rejects_request_frame_outside_recording_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedVideoRecorder(
                root,
                origin=ClockOrigin.capture(),
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            recorder.write(
                _third(0),
                _robot(0),
                overlay=_overlay(0),
                source_elapsed_seconds=1.0,
            )
            recorder.finalize()
            (root / "frame_requests.jsonl").write_text(
                json.dumps(
                    {
                        "record_type": "frame_request",
                        "request_id": "outside",
                        "capture_elapsed_seconds": 50.0,
                    }
                )
                + "\n"
            )
            audit = audit_attempt_videos(root)
            self.assertFalse(audit["passed"])
            self.assertTrue(
                any("outside recording interval" in item for item in audit["errors"])
            )

    def test_audit_rejects_request_pixels_forged_under_a_video_frame_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-forged",
                attempt_number=1,
            )
            video_pixels = _robot(1)
            attempt.frames.record(
                request_id="forged-request",
                logical_agent="Planner",
                payload=_png(_robot(0)),
                camera_view_id="robot_camera",
                frame_sequence_number=1,
                prompt_or_request_body={"task": "controlled mismatch"},
                source_frame_sha256=source_frame_sha256(video_pixels),
                capture_monotonic_ns=attempt.origin.monotonic_ns + 100_000_000,
            )
            video = SynchronizedVideoRecorder(
                attempt.attempt_dir,
                origin=attempt.origin,
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            video.write(
                _third(1),
                video_pixels,
                overlay=_overlay(1),
                source_elapsed_seconds=0.1,
                source_reference=SourceFrameReference(
                    robot_camera_id="robot_camera:1"
                ),
            )
            video.finalize()
            audit = audit_attempt_videos(attempt.attempt_dir)
            self.assertFalse(audit["passed"])
            self.assertTrue(
                any("pixels differ" in item for item in audit["errors"]),
                audit["errors"],
            )

    def test_request_uses_raw_camera_capture_time_after_cfr_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-quantized-request",
                attempt_number=1,
            )
            pixels = _robot(2)
            capture_ns = attempt.origin.monotonic_ns + 237_900_000
            frame = attempt.frames.record(
                request_id="quantized-request",
                logical_agent="Planner",
                payload=_png(pixels),
                camera_view_id="robot_camera",
                frame_sequence_number=2,
                prompt_or_request_body={"task": "quantized capture"},
                source_frame_sha256=source_frame_sha256(pixels),
                capture_monotonic_ns=capture_ns,
            )
            video = SynchronizedVideoRecorder(
                attempt.attempt_dir,
                origin=attempt.origin,
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0.2),
            )
            video.write(
                _third(2),
                pixels,
                overlay=_overlay(2),
                source_elapsed_seconds=0.2379,
                source_reference=SourceFrameReference(
                    robot_camera_id=frame["camera_frame_id"]
                ),
            )
            metadata = video.finalize()
            actual = next(
                item
                for item in metadata["source_frame_linkage"]
                if item["hold_kind"] is None
            )
            self.assertEqual(actual["source_elapsed_seconds"], 0.2)
            self.assertEqual(actual["source_capture_elapsed_seconds"], 0.2379)
            audit = audit_attempt_videos(attempt.attempt_dir)
            self.assertTrue(audit["passed"], audit["errors"])

    def test_overlay_hold_is_ineligible_as_request_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-overlay-request",
                attempt_number=1,
            )
            pixels = _robot(0)
            video = SynchronizedVideoRecorder(
                attempt.attempt_dir,
                origin=attempt.origin,
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            video.write(
                _third(0),
                pixels,
                overlay=_overlay(0),
                source_elapsed_seconds=0.0,
                source_reference=SourceFrameReference(
                    robot_camera_id="robot_camera:0"
                ),
            )
            video.advance_overlay(overlay=_overlay(1))
            metadata = video.finalize()
            overlay_link = next(
                item
                for item in metadata["source_frame_linkage"]
                if item["hold_kind"] == "overlay_update"
            )
            overlay_capture_ns = attempt.origin.monotonic_ns + round(
                float(overlay_link["source_elapsed_seconds"]) * 1_000_000_000
            )
            attempt.frames.record(
                request_id="overlay-only-request",
                logical_agent="Planner",
                payload=_png(pixels),
                camera_view_id="robot_camera",
                frame_sequence_number=0,
                prompt_or_request_body={"task": "must use a true acquisition"},
                source_frame_sha256=source_frame_sha256(pixels),
                capture_monotonic_ns=overlay_capture_ns,
            )
            audit = audit_attempt_videos(attempt.attempt_dir)
            self.assertFalse(audit["passed"])
            self.assertTrue(
                any("not synchronized" in item for item in audit["errors"]),
                audit["errors"],
            )

    def test_lossy_exact_request_retains_lossless_video_source_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-lossy",
                attempt_number=1,
            )
            source_pixels = _robot(2)
            request_jpeg = _jpeg(source_pixels)
            decoded_jpeg = cv2.imdecode(
                np.frombuffer(request_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertNotEqual(
                source_frame_sha256(decoded_jpeg),
                source_frame_sha256(source_pixels),
            )
            attempt.frames.record(
                request_id="sam-request",
                logical_agent="execution_grounding",
                payload=request_jpeg,
                source_payload=_png(source_pixels),
                camera_view_id="robot_camera",
                frame_sequence_number=2,
                prompt_or_request_body={"prompt": "red cube", "quality": 80},
                source_frame_sha256=source_frame_sha256(source_pixels),
                capture_monotonic_ns=attempt.origin.monotonic_ns + 200_000_000,
            )
            video = SynchronizedVideoRecorder(
                attempt.attempt_dir,
                origin=attempt.origin,
                config=VideoConfig(pre_roll_seconds=0, final_hold_seconds=0),
            )
            for step, elapsed in ((1, 0.1), (2, 0.2), (3, 0.3)):
                video.write(
                    _third(step),
                    source_pixels if step == 2 else _robot(step),
                    overlay=_overlay(step),
                    source_elapsed_seconds=elapsed,
                    source_reference=SourceFrameReference(
                        robot_camera_id=f"robot_camera:{step}"
                    ),
                )
            video.finalize()

            audit = audit_attempt_videos(attempt.attempt_dir)

            self.assertTrue(audit["passed"], audit["errors"])

    def test_complete_attempt_passes_recording_and_video_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S0004",
                attempt_number=1,
            )
            attempt.write_setup(
                {
                    "campaign_id": "pilot",
                    "schedule_id": "S0004",
                    "attempt_id": 1,
                    "run_status": "NOT_RUN",
                }
            )
            attempt.write_inputs({"task": "controlled ndarray pilot"})
            attempt.write_initial_state({"step": 0}, _png(_robot(0)))
            attempt.terminal("recording initialized")
            attempt.events.append("scene_reset_complete")
            request_event = attempt.events.append(
                "model_request",
                request_id="req-monitor-001",
                call_id="call-monitor-001",
                logical_agent="Monitor",
            )
            robot_request_png = _png(_robot(1))
            image_data_url = (
                "data:image/png;base64,"
                + base64.b64encode(robot_request_png).decode()
            )
            wire_request = {
                "question": "is the scene safe?",
                "frame_request_id": "req-monitor-001",
                "image_url": {"url": image_data_url},
            }
            frame_request = attempt.frames.record(
                request_id="req-monitor-001",
                logical_agent="Monitor",
                payload=image_data_url,
                camera_view_id="robot_camera",
                frame_sequence_number=0,
                prompt_or_request_body=wire_request,
                source_frame_sha256=source_frame_sha256(_robot(1)),
                capture_monotonic_ns=attempt.origin.monotonic_ns + 300_000_000,
                event_ids=(request_event["event_id"],),
            )
            attempt.model_calls.start(
                call_id="call-monitor-001",
                request_id="req-monitor-001",
                logical_agent="Monitor",
                request=wire_request,
                frame_request_ids=("req-monitor-001",),
                event_id=request_event["event_id"],
            )
            response_event = attempt.events.append(
                "model_response",
                request_id="req-monitor-001",
                call_id="call-monitor-001",
                response_id="resp-monitor-001",
                logical_agent="Monitor",
            )
            attempt.model_calls.end(
                call_id="call-monitor-001",
                response={"safe": True},
                response_id="resp-monitor-001",
                event_id=response_event["event_id"],
            )
            attempt.frames.link_response(
                "req-monitor-001",
                response_id="resp-monitor-001",
                event_ids=(response_event["event_id"],),
            )
            video = SynchronizedVideoRecorder(
                attempt.attempt_dir,
                origin=attempt.origin,
                config=VideoConfig(pre_roll_seconds=0.2, final_hold_seconds=0.2),
            )
            for step, elapsed in enumerate((0.2, 0.3, 0.4)):
                video.write(
                    _third(step),
                    _robot(step),
                    overlay=_overlay(step),
                    source_elapsed_seconds=elapsed,
                    source_reference=SourceFrameReference(
                        robot_camera_id=(
                            frame_request["camera_frame_id"] if step == 1 else f"robot:{step}"
                        )
                    ),
                )
            oracle_event = attempt.events.append(
                "oracle_complete", oracle_verdict="PASS"
            )
            video.finalize()
            attempt.write_final_state({"step": 3}, _png(_robot(2)))
            audit = attempt.finalize(
                run_status=CampaignRunStatus.VALID_PASS,
                oracle_verdict="PASS",
                result={"complete": True},
                diagnosis={
                    "summary": (
                        "Controlled pilot passed "
                        f"[Event {oracle_event['event_id']}]"
                        f"(./experiment_record.md#event-{oracle_event['event_id'].casefold()})."
                    )
                },
                require_final_state=True,
            )
            self.assertTrue(audit["passed"], audit["errors"])
            artifact_checks = json.loads(
                (attempt.attempt_dir / "artifact_checks.json").read_text()
            )
            self.assertTrue(artifact_checks["passed"], artifact_checks["errors"])


if __name__ == "__main__":
    unittest.main()
