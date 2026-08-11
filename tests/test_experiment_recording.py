from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

from experiments.harness.recording import (
    AttemptRecorder,
    ClockOrigin,
    EventJournal,
    RecordingError,
    RunStatus,
    audit_attempt,
    decode_image_payload,
    safe_attempt_directory,
    sha256_bytes,
    verify_checksums,
    write_checksums,
)


def _png(color: tuple[int, int, int] = (10, 80, 220)) -> bytes:
    image = np.full((24, 32, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("test fixture PNG did not encode")
    return encoded.tobytes()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
    )


def _visual_errors(attempt_dir: Path) -> list[str]:
    report = audit_attempt(
        attempt_dir,
        verify_video=False,
        finalizing=True,
    )
    return [
        error
        for error in report["errors"]
        if str(error).startswith("visual provenance:")
    ]


def _successful_visual_exchange(
    campaign: Path,
    *,
    sam_jpeg: bool = False,
    image_in_text_field_only: bool = False,
    redacted_request: bool = False,
) -> tuple[AttemptRecorder, bytes, str]:
    recorder = AttemptRecorder.create(
        campaign,
        schedule_id="S-VISUAL",
        attempt_number=1,
    )
    source_png = _png()
    if sam_jpeg:
        source = cv2.imdecode(
            np.frombuffer(source_png, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if source is None:
            raise AssertionError("test fixture PNG did not decode")
        ok, encoded = cv2.imencode(
            ".jpg", source, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )
        if not ok:
            raise AssertionError("test fixture JPEG did not encode")
        transport_image = encoded.tobytes()
        transport_text = base64.b64encode(transport_image).decode("ascii")
        request: dict[str, object] = {
            "protocol": "multipart/form-data",
            "image": {
                "filename": "frame.jpg",
                "content_type": "image/jpeg",
                "sha256": sha256_bytes(transport_image),
                "bytes_base64": transport_text,
            },
        }
        frame_payload: bytes | str = transport_image
        source_payload: bytes | None = source_png
    else:
        transport_image = source_png
        transport_text = (
            "data:image/png;base64,"
            + base64.b64encode(transport_image).decode("ascii")
        )
        content_type = "image_url"
        content_key = "text" if image_in_text_field_only else "image_url"
        content_value: object = (
            transport_text
            if image_in_text_field_only
            else {"url": transport_text}
        )
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": content_type, content_key: content_value},
                    ],
                }
            ]
        }
        frame_payload = transport_text
        source_payload = None
    if redacted_request:
        request["api_key"] = "fixture-secret"

    request_event = recorder.events.append(
        "model_request",
        request_id="req-visual-001",
        call_id="call-visual-001",
        logical_agent="execution_grounding" if sam_jpeg else "monitor",
    )
    logical_agent = "execution_grounding" if sam_jpeg else "monitor"
    recorder.frames.record(
        request_id="req-visual-001",
        logical_agent=logical_agent,
        payload=frame_payload,
        source_payload=source_payload,
        camera_view_id="robot_camera",
        frame_sequence_number=3,
        prompt_or_request_body=request,
        source_frame_sha256=sha256_bytes(source_png),
        event_ids=(request_event["event_id"],),
    )
    recorder.model_calls.start(
        call_id="call-visual-001",
        request_id="req-visual-001",
        logical_agent=logical_agent,
        request=request,
        model_id="sam3.1" if sam_jpeg else "visual-model",
        frame_request_ids=("req-visual-001",),
        event_id=request_event["event_id"],
    )
    response_event = recorder.events.append(
        "model_response",
        request_id="req-visual-001",
        call_id="call-visual-001",
        response_id="resp-visual-001",
        logical_agent=logical_agent,
    )
    recorder.model_calls.end(
        call_id="call-visual-001",
        response={"ok": True},
        response_id="resp-visual-001",
        event_id=response_event["event_id"],
    )
    recorder.frames.link_response(
        "req-visual-001",
        response_id="resp-visual-001",
        event_ids=(response_event["event_id"],),
    )
    return recorder, transport_image, transport_text


class ExperimentRecordingTests(unittest.TestCase):
    def test_event_journal_thread_safe_stable_ids_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            origin = ClockOrigin.capture()
            path = Path(directory) / "events.jsonl"
            journal = EventJournal(path, origin=origin)
            threads = [
                threading.Thread(
                    target=lambda worker=worker: [
                        journal.append("worker_tick", worker=worker, tick=tick)
                        for tick in range(25)
                    ]
                )
                for worker in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event_id"] for record in records],
                [f"E{number:06d}" for number in range(1, 101)],
            )
            self.assertTrue(
                all(
                    current["monotonic_ns"] <= following["monotonic_ns"]
                    for current, following in zip(records, records[1:])
                )
            )
            resumed = EventJournal(path, origin=origin)
            self.assertEqual(resumed.append("resumed")["event_id"], "E000101")

    def test_attempt_paths_are_new_and_confined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            first = safe_attempt_directory(campaign, "S0001", 1)
            self.assertEqual(
                first, campaign.resolve() / "episodes" / "S0001" / "attempt_1"
            )
            with self.assertRaisesRegex(RecordingError, "refusing to reuse"):
                safe_attempt_directory(campaign, "S0001", 1)
            for unsafe in ("../escape", "/absolute", "has/slash", ".."):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    safe_attempt_directory(campaign, unsafe, 2)

    def test_exact_frames_raw_calls_redaction_and_checksum_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S0002",
                attempt_number=1,
                secret_fields=("experiment_credential",),
                secret_values=("do-not-print",),
            )
            raw_png = _png()
            data_url = "data:image/png;base64," + base64.b64encode(raw_png).decode()
            event = recorder.events.append("model_request_started", logical_agent="HRI")
            frame = recorder.frames.record(
                request_id="req-001",
                logical_agent="HRI",
                payload=data_url,
                camera_view_id="robot_camera",
                frame_sequence_number=7,
                prompt_or_request_body={"prompt": "locate block"},
            )
            recorder.model_calls.start(
                call_id="call-001",
                request_id="req-001",
                logical_agent="HRI",
                request={
                    "messages": [{"role": "user", "content": "locate block"}],
                    "Authorization": "Bearer super-secret",
                    "experiment_credential": "another-secret",
                },
                frame_request_ids=("req-001",),
                event_id=event["event_id"],
            )
            end = recorder.events.append(
                "model_request_finished", response_id="resp-001"
            )
            recorder.model_calls.end(
                call_id="call-001",
                response={"answer": "ok", "password": "response-secret"},
                response_id="resp-001",
                event_id=end["event_id"],
            )
            recorder.frames.link_response(
                "req-001", response_id="resp-001", event_ids=(end["event_id"],)
            )
            recorder.model_calls.start(
                call_id="call-002",
                request_id="req-002",
                logical_agent="Planner",
                request=b'{"api_key":"raw-byte-secret","task":"x"}',
            )
            recorder.model_calls.error(
                call_id="call-002", error=RuntimeError("malformed upstream response")
            )
            recorder.terminal("authorization: Bearer terminal-secret do-not-print")
            recorder.events.append(
                "redaction_probe",
                password="event-secret",
                note="do-not-print",
            )

            saved = recorder.attempt_dir / frame["image_path"]
            self.assertEqual(saved.read_bytes(), raw_png)
            self.assertEqual(frame["raw_image_sha256"], sha256_bytes(raw_png))
            self.assertEqual(
                frame["encoded_request_payload_sha256"],
                sha256_bytes(data_url.encode()),
            )
            all_bytes = b"\n".join(
                path.read_bytes()
                for path in recorder.attempt_dir.rglob("*")
                if path.is_file()
            )
            for secret in (
                b"super-secret",
                b"another-secret",
                b"response-secret",
                b"raw-byte-secret",
                b"terminal-secret",
                b"do-not-print",
                b"event-secret",
            ):
                with self.subTest(secret=secret):
                    self.assertNotIn(secret, all_bytes)
            redactions = [
                json.loads(line)
                for line in (
                    recorder.attempt_dir / "model_calls/redactions.jsonl"
                ).read_text().splitlines()
            ]
            self.assertGreaterEqual(
                sum(item["redaction_count"] for item in redactions), 4
            )

            write_checksums(recorder.attempt_dir)
            checksum = verify_checksums(recorder.attempt_dir)
            self.assertTrue(checksum["passed"], checksum["errors"])
            (recorder.attempt_dir / "terminal.log").write_text("tampered")
            checksum = verify_checksums(recorder.attempt_dir)
            self.assertFalse(checksum["passed"])
            self.assertIn("checksum mismatch: terminal.log", checksum["errors"])

    def test_visual_audit_binds_nested_data_url_and_sam_jpeg_bytes(self) -> None:
        for sam_jpeg, redacted_request in ((False, False), (True, False), (False, True)):
            with self.subTest(
                sam_jpeg=sam_jpeg,
                redacted_request=redacted_request,
            ), tempfile.TemporaryDirectory() as directory:
                recorder, _transport_image, _transport_text = (
                    _successful_visual_exchange(
                        Path(directory) / "campaign",
                        sam_jpeg=sam_jpeg,
                        redacted_request=redacted_request,
                    )
                )
                self.assertEqual(_visual_errors(recorder.attempt_dir), [])

    def test_image_bearing_call_cannot_claim_nonvisual_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-UNDECLARED-VISUAL",
                attempt_number=1,
            )
            raw_png = _png()
            data_url = (
                "data:image/png;base64,"
                + base64.b64encode(raw_png).decode("ascii")
            )
            request = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                        ],
                    }
                ]
            }
            started = recorder.events.append(
                "model_request",
                request_id="req-undeclared-001",
                call_id="call-undeclared-001",
                logical_agent="monitor",
            )
            recorder.model_calls.start(
                call_id="call-undeclared-001",
                request_id="req-undeclared-001",
                logical_agent="monitor",
                request=request,
                frame_request_ids=(),
                event_id=started["event_id"],
            )
            completed = recorder.events.append(
                "model_response",
                request_id="req-undeclared-001",
                call_id="call-undeclared-001",
                response_id="resp-undeclared-001",
                logical_agent="monitor",
            )
            recorder.model_calls.end(
                call_id="call-undeclared-001",
                response={"ok": True},
                response_id="resp-undeclared-001",
                event_id=completed["event_id"],
            )

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("declares no frame_request_ids" in error for error in errors),
                errors,
            )

    def test_visual_audit_does_not_accept_data_url_in_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign",
                image_in_text_field_only=True,
            )
            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("does not contain the exact transport image" in error for error in errors),
                errors,
            )

    def test_visual_audit_catches_request_artifact_and_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            calls_path = recorder.attempt_dir / "model_calls.jsonl"
            call_records = _read_jsonl(calls_path)
            request_artifact = recorder.attempt_dir / str(
                call_records[0]["request_artifact"]
            )
            request_value = json.loads(request_artifact.read_text())
            replacement = (
                "data:image/png;base64,"
                + base64.b64encode(_png((220, 30, 10))).decode("ascii")
            )
            self.assertNotEqual(transport_text, replacement)
            request_value["messages"][0]["content"][0]["image_url"]["url"] = (
                replacement
            )
            request_artifact.write_text(
                json.dumps(request_value, indent=2, sort_keys=True) + "\n"
            )
            call_records[0]["request_sha256"] = sha256_bytes(
                request_artifact.read_bytes()
            )
            _write_jsonl(calls_path, call_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("does not contain the exact transport image" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("prompt/request-body digest" in error for error in errors),
                errors,
            )

    def test_visual_audit_rejects_duplicate_payload_for_one_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            calls_path = recorder.attempt_dir / "model_calls.jsonl"
            call_records = _read_jsonl(calls_path)
            request_artifact = recorder.attempt_dir / str(
                call_records[0]["request_artifact"]
            )
            request_value = json.loads(request_artifact.read_text())
            content = request_value["messages"][0]["content"]
            content.append(dict(content[0]))
            request_artifact.write_text(
                json.dumps(request_value, indent=2, sort_keys=True) + "\n"
            )
            call_records[0]["request_sha256"] = sha256_bytes(
                request_artifact.read_bytes()
            )
            _write_jsonl(calls_path, call_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("cardinality mismatch" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("duplicate visual payload" in error for error in errors),
                errors,
            )

    def test_visual_audit_rejects_forged_redaction_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            calls_path = recorder.attempt_dir / "model_calls.jsonl"
            call_records = _read_jsonl(calls_path)
            request_artifact = recorder.attempt_dir / str(
                call_records[0]["request_artifact"]
            )
            request_value = json.loads(request_artifact.read_text())
            request_value["instruction"] = "tampered after recording"
            request_artifact.write_text(
                json.dumps(request_value, indent=2, sort_keys=True) + "\n"
            )
            call_records[0]["request_sha256"] = sha256_bytes(
                request_artifact.read_bytes()
            )
            call_records[0]["redaction_count"] = 1
            _write_jsonl(calls_path, call_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("not bound to exactly one" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("prompt/request-body digest" in error for error in errors),
                errors,
            )

    def test_visual_audit_catches_tamper_outside_real_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign",
                redacted_request=True,
            )
            calls_path = recorder.attempt_dir / "model_calls.jsonl"
            call_records = _read_jsonl(calls_path)
            request_artifact = recorder.attempt_dir / str(
                call_records[0]["request_artifact"]
            )
            request_value = json.loads(request_artifact.read_text())
            self.assertEqual(request_value["api_key"], "[REDACTED]")
            request_value["instruction"] = "tampered outside the redacted path"
            request_artifact.write_text(
                json.dumps(request_value, indent=2, sort_keys=True) + "\n"
            )
            call_records[0]["request_sha256"] = sha256_bytes(
                request_artifact.read_bytes()
            )
            _write_jsonl(calls_path, call_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("frame-ledger commitment" in error for error in errors),
                errors,
            )

    def test_visual_audit_catches_relabelled_frame_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            calls_path = recorder.attempt_dir / "model_calls.jsonl"
            call_records = _read_jsonl(calls_path)
            call_records[0]["request_id"] = "req-forged"
            call_records[0]["frame_request_ids"] = ["req-forged"]
            _write_jsonl(calls_path, call_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("references unknown frame request req-forged" in error for error in errors),
                errors,
            )
            self.assertTrue(
                any("belongs to 0 model call starts" in error for error in errors),
                errors,
            )

    def test_visual_audit_binds_response_id_and_end_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            frames_path = recorder.attempt_dir / "frame_requests.jsonl"
            frame_records = _read_jsonl(frames_path)
            frame_records[-1]["response_id"] = "resp-forged"
            _write_jsonl(frames_path, frame_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("disagree on response_id" in error for error in errors),
                errors,
            )

        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            events_path = recorder.attempt_dir / "events.jsonl"
            event_records = _read_jsonl(events_path)
            event_records[-1]["event_type"] = "model_request"
            _write_jsonl(events_path, event_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("response event is not model_response" in error for error in errors),
                errors,
            )

        with tempfile.TemporaryDirectory() as directory:
            recorder, _transport_image, _transport_text = _successful_visual_exchange(
                Path(directory) / "campaign"
            )
            events_path = recorder.attempt_dir / "events.jsonl"
            event_records = _read_jsonl(events_path)
            event_records[0]["event_type"] = "model_response"
            _write_jsonl(events_path, event_records)

            errors = _visual_errors(recorder.attempt_dir)
            self.assertTrue(
                any("start event is not model_request" in error for error in errors),
                errors,
            )

    def test_nonvisual_model_call_does_not_require_image_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-EMBEDDING",
                attempt_number=1,
            )
            started = recorder.events.append(
                "model_request_started",
                request_id="req-embedding-001",
                call_id="call-embedding-001",
                logical_agent="memory_embedding",
            )
            recorder.model_calls.start(
                call_id="call-embedding-001",
                request_id="req-embedding-001",
                logical_agent="memory_embedding",
                request={"input": ["document"]},
                frame_request_ids=(),
                event_id=started["event_id"],
            )
            completed = recorder.events.append(
                "model_response_received",
                request_id="req-embedding-001",
                call_id="call-embedding-001",
                response_id="resp-embedding-001",
                logical_agent="memory_embedding",
            )
            recorder.model_calls.end(
                call_id="call-embedding-001",
                response={"embedding": [0.1, 0.2]},
                response_id="resp-embedding-001",
                event_id=completed["event_id"],
            )
            self.assertEqual(_visual_errors(recorder.attempt_dir), [])

    def test_invalid_image_is_rejected_instead_of_placeholder(self) -> None:
        with self.assertRaisesRegex(RecordingError, "does not decode"):
            decode_image_payload(b"not an image")
        with self.assertRaisesRegex(RecordingError, "strict base64"):
            decode_image_payload("data:image/png;base64,not!!base64")

    def test_context_exception_preserves_partial_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder: AttemptRecorder | None = None
            with self.assertRaisesRegex(RuntimeError, "controlled crash"):
                with AttemptRecorder.create(
                    Path(directory) / "campaign",
                    schedule_id="S0003",
                    attempt_number=1,
                ) as active:
                    recorder = active
                    active.write_setup(
                        {"campaign_id": "pilot", "run_status": "NOT_RUN"}
                    )
                    active.write_inputs({"task": "pilot"})
                    active.write_initial_state({"objects": []}, _png())
                    active.events.append("controlled_crash_about_to_fire")
                    raise RuntimeError("controlled crash")
            if recorder is None:
                self.fail("attempt recorder fixture was not created")
            self.assertTrue((recorder.attempt_dir / "error.json").is_file())
            self.assertTrue((recorder.attempt_dir / "results.json").is_file())
            self.assertTrue((recorder.attempt_dir / "experiment_record.md").is_file())
            self.assertTrue((recorder.attempt_dir / "diagnosis.json").is_file())
            self.assertTrue((recorder.attempt_dir / "checksums.sha256").is_file())
            state = json.loads(
                (recorder.attempt_dir / "recording_state.json").read_text()
            )
            self.assertEqual(state["run_status"], RunStatus.INVALID_HARNESS.value)
            checks = json.loads(
                (recorder.attempt_dir / "artifact_checks.json").read_text()
            )
            self.assertFalse(checks["passed"])
            self.assertTrue(any("video.mp4" in item for item in checks["errors"]))


if __name__ == "__main__":
    unittest.main()
