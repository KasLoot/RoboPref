from __future__ import annotations

import socket
import threading
import unittest
from unittest.mock import MagicMock, patch

from prefmem.stream_camera import WebcamServer
from prefmem.task_publisher import (
    CameraTaskPublisher,
    ControllerDisplay,
    DisplayState,
    TaskPublisherConnectionError,
    TaskPublisherProtocolError,
    TaskPublisherResponseError,
)


class FakeCamera:
    def health(self) -> tuple[dict[str, object], bool]:
        return {"status": "ok"}, True


def active_display(*, sequence: int = 1) -> ControllerDisplay:
    return ControllerDisplay(
        session_id="run-1",
        sequence=sequence,
        state=DisplayState.ACTIVE,
        goal="Build a stable tower.",
        cycle=1,
        publication_id="run-1:cycle-1:attempt-1",
        instruction="Place the red block flat in the marked area.",
        expected_observation=(
            "The red block is flat and stable in the marked area.",
        ),
    )


class ControllerDisplayContractTests(unittest.TestCase):
    def test_planning_cannot_retain_retired_execution_instruction(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot retain"):
            ControllerDisplay(
                session_id="run-1",
                sequence=2,
                state=DisplayState.PLANNING,
                goal="Build a stable tower.",
                instruction="Continue stacking.",
            )

    def test_active_requires_full_single_task_projection(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires publication_id"):
            ControllerDisplay(
                session_id="run-1",
                sequence=1,
                state=DisplayState.ACTIVE,
                goal="Build a stable tower.",
                cycle=1,
                instruction="Place the first block.",
                expected_observation=("The first block is stable.",),
            )

    def test_attention_and_emergency_require_explanation(self) -> None:
        for state in (
            DisplayState.NEEDS_ATTENTION,
            DisplayState.EMERGENCY_STOPPED,
        ):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "requires message"):
                    ControllerDisplay(
                        session_id="run-1",
                        sequence=1,
                        state=state,
                        goal="Build a stable tower.",
                    )

    def test_non_execution_states_cannot_present_an_execution_instruction(self) -> None:
        for state in (
            DisplayState.FINAL_VALIDATION,
            DisplayState.COMPLETE,
            DisplayState.EMERGENCY_STOPPED,
        ):
            with self.subTest(state=state):
                kwargs: dict[str, object] = {}
                if state is DisplayState.FINAL_VALIDATION:
                    kwargs = {
                        "publication_id": "run-1:final-1",
                        "expected_observation": ("The goal is satisfied.",),
                    }
                if state is DisplayState.EMERGENCY_STOPPED:
                    kwargs["message"] = "Emergency stop activated."
                with self.assertRaisesRegex(ValueError, "execution instruction"):
                    ControllerDisplay(
                        session_id="run-1",
                        sequence=1,
                        state=state,
                        goal="Build a stable tower.",
                        instruction="Continue moving blocks.",
                        **kwargs,
                    )

    def test_every_surface_state_round_trips(self) -> None:
        displays = (
            ControllerDisplay(
                session_id="run-1",
                sequence=1,
                state=DisplayState.PLANNING,
                goal="Build a stable tower.",
            ),
            active_display(sequence=2),
            ControllerDisplay(
                session_id="run-1",
                sequence=3,
                state=DisplayState.FINAL_VALIDATION,
                goal="Build a stable tower.",
                publication_id="run-1:final-1",
                expected_observation=("All blocks form one stable tower.",),
            ),
            ControllerDisplay(
                session_id="run-1",
                sequence=4,
                state=DisplayState.NEEDS_ATTENTION,
                goal="Build a stable tower.",
                message="The camera view is occluded.",
            ),
            ControllerDisplay(
                session_id="run-1",
                sequence=5,
                state=DisplayState.COMPLETE,
                goal="Build a stable tower.",
                monitor_observation="All blocks form one stable tower.",
            ),
            ControllerDisplay(
                session_id="run-1",
                sequence=6,
                state=DisplayState.EMERGENCY_STOPPED,
                goal="Build a stable tower.",
                message="A person entered the robot workspace.",
            ),
        )

        for display in displays:
            with self.subTest(state=display.state):
                self.assertEqual(
                    ControllerDisplay.from_payload(display.to_payload()),
                    display,
                )


class CameraTaskPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = WebcamServer(("127.0.0.1", 0), FakeCamera())
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)

    def test_publish_maps_typed_display_and_returns_acknowledgement(self) -> None:
        publisher = CameraTaskPublisher(self.base_url)
        display = active_display()

        envelope = publisher.publish(display)

        self.assertEqual(envelope.revision, 1)
        self.assertEqual(envelope.display, display)
        self.assertEqual(
            self.server.display_store.snapshot()["display"],
            display.to_payload(),
        )

    def test_server_conflict_is_exposed_with_status_and_reason(self) -> None:
        publisher = CameraTaskPublisher(self.base_url)
        publisher.publish(active_display(sequence=2))

        with self.assertRaises(TaskPublisherResponseError) as raised:
            publisher.publish(active_display(sequence=1))

        self.assertEqual(raised.exception.status, 409)
        self.assertIn("older", raised.exception.message)

    def test_reset_uses_session_ownership_and_retires_it(self) -> None:
        publisher = CameraTaskPublisher(self.base_url)
        publisher.publish(active_display())

        envelope = publisher.reset(session_id="run-1")

        self.assertEqual(envelope.revision, 2)
        self.assertIsNone(envelope.display)
        with self.assertRaises(TaskPublisherResponseError) as raised:
            publisher.publish(active_display(sequence=2))
        self.assertEqual(raised.exception.status, 409)
        self.assertIn("retired", raised.exception.message)

    def test_client_rejects_non_loopback_base_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            CameraTaskPublisher("http://192.168.1.25:1234")
        with self.assertRaisesRegex(ValueError, "must use http"):
            CameraTaskPublisher("https://127.0.0.1:1234")

    def test_connection_failure_has_a_distinct_error(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        publisher = CameraTaskPublisher(f"http://127.0.0.1:{port}", timeout=0.2)

        with self.assertRaises(TaskPublisherConnectionError):
            publisher.publish(active_display())

    def test_invalid_acknowledgement_has_a_distinct_error(self) -> None:
        publisher = CameraTaskPublisher(self.base_url)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"unexpected":true}'

        with patch.object(publisher._opener, "open", return_value=response):
            with self.assertRaises(TaskPublisherProtocolError):
                publisher.publish(active_display())


if __name__ == "__main__":
    unittest.main()
