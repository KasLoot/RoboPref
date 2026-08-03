from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
import unittest

from langchain.messages import HumanMessage, SystemMessage

from prefmem.agents.monitor import (
    CapturedFrame,
    MonitorErrorKind,
    MonitorModelOutput,
    MonitorOutputError,
    MonitorService,
    THINKING_DISABLED_OPTIONS,
)
from prefmem.contracts import (
    ObservationCriterion,
    PublishedTask,
    TaskPhase,
    TaskStatus,
)
from prefmem.emergency import EmergencyStopCoordinator


def task(
    publication_id: str = "goal-1:cycle-1:attempt-1",
    *,
    frame_sequence: int = 4,
) -> PublishedTask:
    return PublishedTask(
        plan_id="goal-1",
        revision=1,
        step_id="goal-1:cycle-1:task",
        phase=TaskPhase.STEP,
        instruction="Place the blue block flat in the marked zone.",
        expected_observation=(
            ObservationCriterion(
                criterion_id="goal-1:cycle-1:c1",
                description="The blue block is flat in the marked zone.",
            ),
        ),
        known_failure_conditions=(
            "The blue block has fallen off the table.",
        ),
        published_at=1.0,
        publication_id=publication_id,
        cycle_id=1,
        frame_sequence=frame_sequence,
    )


def normal_output(status: str = "SUCCESS") -> str:
    state = "MET" if status == "SUCCESS" else "NOT_MET"
    failure = None
    if status == "FAIL":
        failure = {
            "kind": "UNEXPECTED",
            "description": "The block visibly fell off the table.",
        }
    return json.dumps(
        {
            "emergency_stop": False,
            "emergency_reason": None,
            "task_status": status,
            "criteria": [
                {"id": "goal-1:cycle-1:c1", "state": state}
            ],
            "failure": failure,
            "observation": "The current frame provides decisive visible evidence.",
        }
    )


class SequenceFrames:
    def __init__(self, sequences: list[int]) -> None:
        self.sequences = list(sequences)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> CapturedFrame:
        with self._lock:
            index = min(self.calls, len(self.sequences) - 1)
            sequence = self.sequences[index]
            self.calls += 1
        return CapturedFrame(
            image_block={
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/"},
            },
            observed_at=10.0 + self.calls,
            sequence=sequence,
        )


class RecordingModel:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.output)


class MonitorOutputTests(unittest.TestCase):
    def test_outer_contract_is_strict_and_emergency_fields_are_mandatory(self) -> None:
        with self.assertRaisesRegex(MonitorOutputError, "missing fields"):
            MonitorModelOutput.from_json(
                '{"task_status":"ONGOING","criteria":[],"failure":null,'
                '"observation":"Visible."}'
            )
        with self.assertRaisesRegex(MonitorOutputError, "unknown fields"):
            MonitorModelOutput.from_json(
                '{"emergency_stop":false,"emergency_reason":null,'
                '"task_status":"ONGOING","criteria":[],"failure":null,'
                '"observation":"Visible.","confidence":0.7}'
            )
        with self.assertRaisesRegex(MonitorOutputError, "duplicate field"):
            MonitorModelOutput.from_json(
                '{"emergency_stop":false,"emergency_stop":false,'
                '"emergency_reason":null,"task_status":"ONGOING",'
                '"criteria":[],"failure":null,"observation":"Visible."}'
            )
        with self.assertRaisesRegex(MonitorOutputError, "must be null"):
            MonitorModelOutput.from_json(
                '{"emergency_stop":false,"emergency_reason":"danger",'
                '"task_status":"ONGOING","criteria":[],"failure":null,'
                '"observation":"Visible."}'
            )


class MonitorServiceTests(unittest.TestCase):
    def test_model_order_thinking_disabled_and_host_envelope(self) -> None:
        model = RecordingModel(normal_output())
        frames = SequenceFrames([5])
        received = []
        finished = threading.Event()
        service_holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            service_holder["service"].retire()
            finished.set()

        service = MonitorService(
            accept,
            model=model,
            frame_source=frames,
            system_prompt="Monitor system contract.",
            min_interval_seconds=0,
        )
        service_holder["service"] = service
        try:
            service.publish(task())
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(len(received), 1)
        assessment = received[0]
        self.assertIs(assessment.task_status, TaskStatus.SUCCESS)
        self.assertEqual(
            assessment.publication_id,
            "goal-1:cycle-1:attempt-1",
        )
        self.assertEqual(assessment.frame_sequence, 5)
        messages, kwargs = model.calls[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content[0]["type"], "image_url")
        self.assertEqual(messages[1].content[1]["type"], "text")
        task_text = messages[1].content[1]["text"]
        self.assertIn(
            "Place the blue block flat in the marked zone.",
            task_text,
        )
        self.assertIn("goal-1:cycle-1:c1", task_text)
        self.assertEqual(kwargs, THINKING_DISABLED_OPTIONS)

    def test_prepublication_frame_is_not_sent_to_model(self) -> None:
        model = RecordingModel(normal_output())
        frames = SequenceFrames([4, 6])
        finished = threading.Event()
        service_holder = {}

        def accept(assessment) -> None:
            service_holder["service"].retire()
            finished.set()

        service = MonitorService(
            accept,
            model=model,
            frame_source=frames,
            system_prompt="Monitor system contract.",
            min_interval_seconds=0,
        )
        service_holder["service"] = service
        try:
            service.publish(task(frame_sequence=4))
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertGreaterEqual(frames.calls, 2)
        self.assertEqual(len(model.calls), 1)

    def test_replaced_publication_fences_in_flight_normal_result(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        second_finished = threading.Event()
        active = 0
        max_active = 0
        lock = threading.Lock()

        class BlockingModel:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(inner_self, messages, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    inner_self.calls += 1
                    call = inner_self.calls
                try:
                    if call == 1:
                        first_started.set()
                        release_first.wait(2.0)
                    return SimpleNamespace(content=normal_output("ONGOING"))
                finally:
                    with lock:
                        active -= 1

        model = BlockingModel()
        received = []
        service_holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            service_holder["service"].retire()
            second_finished.set()

        service = MonitorService(
            accept,
            model=model,
            frame_source=SequenceFrames([10, 11, 12]),
            system_prompt="Monitor system contract.",
            min_interval_seconds=0,
        )
        service_holder["service"] = service
        try:
            service.publish(task("publication-old"))
            self.assertTrue(first_started.wait(2.0))
            service.publish(task("publication-new"))
            release_first.set()
            self.assertTrue(second_finished.wait(2.0))
        finally:
            release_first.set()
            service.stop()

        self.assertEqual(max_active, 1)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].publication_id, "publication-new")

    def test_one_emergency_result_bypasses_ordinary_validation_and_latches(self) -> None:
        output = json.dumps(
            {
                "emergency_stop": True,
                "emergency_reason": "A person is inside the robot workspace.",
                # These ordinary fields are intentionally invalid. Emergency
                # must be evaluated first and never enter temporal aggregation.
                "task_status": "INVALID",
                "criteria": "invalid",
                "failure": {"invalid": True},
                "observation": None,
            }
        )
        stop_calls = []
        emergency = EmergencyStopCoordinator(stop_calls.append)
        received = []
        errors = []
        service = MonitorService(
            received.append,
            on_error=errors.append,
            emergency=emergency,
            model=RecordingModel(output),
            frame_source=SequenceFrames([8]),
            system_prompt="Monitor system contract.",
            min_interval_seconds=0,
        )
        try:
            service.publish(task())
            self.assertTrue(service.shutdown_event.wait(2.0))
            deadline = time.monotonic() + 2.0
            while service.running and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            service.stop()

        self.assertEqual(stop_calls, ["A person is inside the robot workspace."])
        self.assertEqual(received, [])
        self.assertEqual(errors, [])
        self.assertEqual(
            emergency.event.publication_id,
            "goal-1:cycle-1:attempt-1",
        )

    def test_model_failure_is_a_system_error_not_task_fail(self) -> None:
        class BrokenModel:
            def invoke(self, messages, **kwargs):
                raise RuntimeError("cloud model unavailable")

        received = []
        errors = []
        finished = threading.Event()
        service_holder = {}

        def report(error) -> None:
            errors.append(error)
            service_holder["service"].retire()
            finished.set()

        service = MonitorService(
            received.append,
            on_error=report,
            model=BrokenModel(),
            frame_source=SequenceFrames([9]),
            system_prompt="Monitor system contract.",
            min_interval_seconds=0,
        )
        service_holder["service"] = service
        try:
            service.publish(task())
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(received, [])
        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0].kind, MonitorErrorKind.MODEL)


if __name__ == "__main__":
    unittest.main()
