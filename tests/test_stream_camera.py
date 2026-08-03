from __future__ import annotations

import http.client
import json
import threading
import unittest
from unittest.mock import patch

from prefmem.stream_camera import (
    PAGE,
    ControllerDisplayStore,
    DisplayConflictError,
    WebcamServer,
    backend_id,
    build_parser,
    configure_capture,
    fourcc_name,
    is_loopback_address,
    validate_display_payload,
)
from prefmem.task_publisher import (
    ControllerDisplay,
    DisplayState,
    FRAME_SEQUENCE_HEADER,
)


class FakeCv2:
    CAP_ANY = 0
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BUFFERSIZE = 38

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        return sum(
            ord(character) << (8 * index)
            for index, character in enumerate(characters)
        )


class FakeCapture:
    def __init__(self) -> None:
        self.settings: list[tuple[int, int]] = []

    def set(self, property_id: int, value: int) -> bool:
        self.settings.append((property_id, value))
        return True


class FakeHttpCamera:
    def health(self) -> tuple[dict[str, object], bool]:
        return {"status": "ok"}, True

    def snapshot_with_sequence(self) -> tuple[int, bytes]:
        return 42, b"jpeg-data"


def active_display(
    *,
    session_id: str = "session-1",
    sequence: int = 1,
    instruction: str = "Place the book in the left zone.",
) -> ControllerDisplay:
    return ControllerDisplay(
        session_id=session_id,
        sequence=sequence,
        state=DisplayState.ACTIVE,
        goal="Put away the objects.",
        cycle=1,
        publication_id=f"{session_id}:cycle-1",
        instruction=instruction,
        expected_observation=("The book is resting in the left zone.",),
        monitor_observation="The book is still being held.",
    )


class StreamCameraConfigurationTests(unittest.TestCase):
    def test_system_flag_accepts_linux_and_windows(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.parse_args(["--system", "linux"]).system, "linux")
        self.assertEqual(parser.parse_args(["--system", "windows"]).system, "windows")

    def test_linux_uses_v4l2_and_requests_mjpeg_first(self) -> None:
        capture = FakeCapture()
        mjpg = FakeCv2.VideoWriter_fourcc(*"MJPG")

        configure_capture(
            FakeCv2,
            capture,
            system="linux",
            width=1280,
            height=720,
            fps=30,
        )

        self.assertEqual(backend_id(FakeCv2, "linux", "auto"), FakeCv2.CAP_V4L2)
        self.assertEqual(
            capture.settings,
            [
                (FakeCv2.CAP_PROP_FOURCC, mjpg),
                (FakeCv2.CAP_PROP_FRAME_WIDTH, 1280),
                (FakeCv2.CAP_PROP_FRAME_HEIGHT, 720),
                (FakeCv2.CAP_PROP_FPS, 30),
                (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
            ],
        )
        self.assertEqual(fourcc_name(mjpg), "MJPG")

    def test_windows_preserves_existing_capture_configuration(self) -> None:
        capture = FakeCapture()

        configure_capture(
            FakeCv2,
            capture,
            system="windows",
            width=1280,
            height=720,
            fps=30,
        )

        self.assertEqual(
            backend_id(FakeCv2, "windows", "dshow"),
            FakeCv2.CAP_DSHOW,
        )
        self.assertEqual(
            capture.settings,
            [
                (FakeCv2.CAP_PROP_FRAME_WIDTH, 1280),
                (FakeCv2.CAP_PROP_FRAME_HEIGHT, 720),
                (FakeCv2.CAP_PROP_FPS, 30),
                (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
            ],
        )


class ControllerDisplayStoreTests(unittest.TestCase):
    def test_publish_validates_and_copies_display(self) -> None:
        store = ControllerDisplayStore()
        payload = active_display().to_payload()

        published = store.publish(payload)
        payload["expected_observation"].append("Mutated by caller")
        published["display"]["expected_observation"].append("Mutated response")

        self.assertEqual(published["revision"], 1)
        self.assertEqual(store.snapshot()["display"], active_display().to_payload())

    def test_publish_is_thread_safe_and_revision_is_monotonic(self) -> None:
        store = ControllerDisplayStore()
        threads = [
            threading.Thread(
                target=store.publish,
                args=(
                    active_display(
                        session_id=f"session-{index}",
                        instruction=f"Task {index}",
                    ).to_payload(),
                ),
            )
            for index in range(20)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(store.snapshot()["revision"], 20)

    def test_validation_rejects_unknown_fields_and_incomplete_active(self) -> None:
        payload = active_display().to_payload()
        payload["extra"] = True
        with self.assertRaisesRegex(ValueError, "unknown display field"):
            validate_display_payload(payload)
        with self.assertRaisesRegex(ValueError, "requires publication_id"):
            validate_display_payload(
                {
                    "session_id": "session-1",
                    "sequence": 1,
                    "state": "ACTIVE",
                    "goal": "Put away the objects.",
                    "cycle": 1,
                    "instruction": "Do it.",
                    "expected_observation": ["It is done."],
                }
            )

    def test_stale_sequence_is_rejected_but_retry_is_idempotent(self) -> None:
        store = ControllerDisplayStore()
        current = active_display(sequence=2).to_payload()
        first = store.publish(current)

        self.assertEqual(store.publish(current), first)
        with self.assertRaisesRegex(DisplayConflictError, "older"):
            store.publish(active_display(sequence=1).to_payload())
        with self.assertRaisesRegex(DisplayConflictError, "reused"):
            store.publish(
                active_display(sequence=2, instruction="Different task.").to_payload()
            )
        self.assertEqual(store.snapshot()["revision"], 1)

    def test_emergency_is_terminal_within_session(self) -> None:
        store = ControllerDisplayStore()
        emergency = ControllerDisplay(
            session_id="session-1",
            sequence=4,
            state=DisplayState.EMERGENCY_STOPPED,
            goal="Put away the objects.",
            message="A person entered the robot workspace.",
        )
        store.publish(emergency.to_payload())

        with self.assertRaisesRegex(DisplayConflictError, "terminal"):
            store.publish(active_display(sequence=5).to_payload())
        self.assertEqual(store.snapshot()["display"], emergency.to_payload())

        replacement = active_display(session_id="session-2").to_payload()
        self.assertEqual(store.publish(replacement)["display"], replacement)
        with self.assertRaisesRegex(DisplayConflictError, "retired"):
            store.publish(
                active_display(session_id="session-1", sequence=6).to_payload()
            )

    def test_complete_persists_until_new_session_or_reset(self) -> None:
        store = ControllerDisplayStore()
        complete = ControllerDisplay(
            session_id="session-1",
            sequence=4,
            state=DisplayState.COMPLETE,
            goal="Put away the objects.",
            monitor_observation="All objects are in their requested zones.",
        )
        store.publish(complete.to_payload())

        with self.assertRaisesRegex(DisplayConflictError, "COMPLETE is terminal"):
            store.publish(active_display(sequence=5).to_payload())
        self.assertEqual(store.snapshot()["display"], complete.to_payload())

    def test_reset_retires_session_and_checks_ownership(self) -> None:
        store = ControllerDisplayStore()
        store.publish(active_display().to_payload())

        with self.assertRaisesRegex(DisplayConflictError, "does not own"):
            store.reset(session_id="another-session")
        reset = store.reset(session_id="session-1")

        self.assertEqual(reset, {"revision": 2, "display": None})
        with self.assertRaisesRegex(DisplayConflictError, "retired"):
            store.publish(active_display(sequence=2).to_payload())

    def test_loopback_detection_supports_ipv4_ipv6_and_mapped_ipv4(self) -> None:
        self.assertTrue(is_loopback_address("127.0.0.1"))
        self.assertTrue(is_loopback_address("::1"))
        self.assertTrue(is_loopback_address("::ffff:127.0.0.1"))
        self.assertFalse(is_loopback_address("192.168.1.4"))


class TaskHttpEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = WebcamServer(("127.0.0.1", 0), FakeHttpCamera())
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object] | bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            response_body = response.read()
            if response.getheader("Content-Type", "").startswith("application/json"):
                return response.status, json.loads(response_body)
            return response.status, response_body
        finally:
            connection.close()

    def test_page_places_task_card_last_and_polls_safely(self) -> None:
        feed_position = PAGE.index(b'id="feed"')
        footer_position = PAGE.index(b"<footer>")
        task_position = PAGE.index(b'id="task-card"')

        self.assertLess(feed_position, footer_position)
        self.assertLess(footer_position, task_position)
        self.assertIn(b'fetch("/api/task"', PAGE)
        self.assertIn(b"window.setTimeout(refreshDisplay, 1000)", PAGE)
        self.assertNotIn(b"setInterval", PAGE)
        self.assertNotIn(b"innerHTML", PAGE)
        self.assertIn(b"Task display unavailable.", PAGE)
        for state in DisplayState:
            self.assertIn(state.value.encode(), PAGE)

    def test_get_starts_empty_and_put_publishes_display(self) -> None:
        status, initial = self.request("GET", "/api/task")
        self.assertEqual(status, 200)
        self.assertEqual(initial, {"revision": 0, "display": None})

        display = active_display().to_payload()
        encoded = json.dumps(display).encode()
        status, published = self.request(
            "PUT",
            "/api/task",
            encoded,
            {"Content-Type": "application/json"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(published, {"revision": 1, "display": display})
        self.assertEqual(self.request("GET", "/api/task"), (200, published))

    def test_put_rejects_invalid_payload_without_replacing_display(self) -> None:
        payload = active_display().to_payload()
        payload["unknown"] = 1

        status, response = self.request(
            "PUT",
            "/api/task",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )

        self.assertEqual(status, 400)
        self.assertIn("unknown display field", response["error"])
        self.assertEqual(
            self.request("GET", "/api/task"),
            (200, {"revision": 0, "display": None}),
        )

    def test_put_rejects_stale_sequence_with_conflict(self) -> None:
        current = active_display(sequence=2).to_payload()
        for payload in (current, active_display(sequence=1).to_payload()):
            status, response = self.request(
                "PUT",
                "/api/task",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )

        self.assertEqual(status, 409)
        self.assertIn("older", response["error"])
        self.assertEqual(self.server.display_store.snapshot()["display"], current)

    def test_put_and_delete_reject_non_loopback_client(self) -> None:
        encoded = json.dumps(active_display().to_payload()).encode()

        with patch(
            "prefmem.stream_camera.is_loopback_address",
            return_value=False,
        ):
            put_status, put_response = self.request(
                "PUT",
                "/api/task",
                encoded,
                {"Content-Type": "application/json"},
            )
            delete_status, delete_response = self.request("DELETE", "/api/task")

        self.assertEqual(put_status, 403)
        self.assertIn("loopback", put_response["error"])
        self.assertEqual(delete_status, 403)
        self.assertIn("loopback", delete_response["error"])
        self.assertEqual(self.server.display_store.snapshot()["revision"], 0)

    def test_delete_clears_and_retires_owned_session(self) -> None:
        display = active_display().to_payload()
        self.request(
            "PUT",
            "/api/task",
            json.dumps(display).encode(),
            {"Content-Type": "application/json"},
        )

        status, reset = self.request(
            "DELETE",
            "/api/task?session_id=session-1",
        )
        stale_status, stale = self.request(
            "PUT",
            "/api/task",
            json.dumps(active_display(sequence=2).to_payload()).encode(),
            {"Content-Type": "application/json"},
        )

        self.assertEqual((status, reset), (200, {"revision": 2, "display": None}))
        self.assertEqual(stale_status, 409)
        self.assertIn("retired", stale["error"])

    def test_snapshot_includes_atomic_frame_sequence_header(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request("GET", "/snapshot.jpg")
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader(FRAME_SEQUENCE_HEADER), "42")
        self.assertEqual(body, b"jpeg-data")


if __name__ == "__main__":
    unittest.main()
