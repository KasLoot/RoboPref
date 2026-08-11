from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage

from prefmem.agents.config import HRI_Config, Memory_Config
from prefmem.agents.hri import HRI_Agent
from prefmem.agents.metrics import MetricsRecordingError, TurnMetrics
from prefmem.cli import parse_args
from prefmem.controller import RecedingHorizonController
from prefmem.execution.contracts import (
    ExecutionErrorKind,
    ExecutionEvent,
    ExecutionState,
)


class FakeBoundModel:
    def bind_tools(self, _tools):
        return self


class ExperimentSourceSeamTests(unittest.TestCase):
    def test_model_and_embedding_endpoints_are_explicitly_overridable(self):
        hri = HRI_Config(
            "vllm",
            model="model-b",
            model_base_url="http://127.0.0.1:8123/v1/",
        )
        memory = Memory_Config(
            "vllm",
            embedding_model="embedding-b",
            embedding_model_base_url="http://127.0.0.1:8124/v1/",
        )

        self.assertEqual(hri.model, "model-b")
        self.assertEqual(hri.model_base_url, "http://127.0.0.1:8123/v1")
        self.assertEqual(memory.embedding_model, "embedding-b")
        self.assertEqual(
            memory.embedding_model_base_url,
            "http://127.0.0.1:8124/v1",
        )

    def test_hri_accepts_injected_role_boundaries_and_instance_identity(self):
        planner = object()
        memory = object()
        args = SimpleNamespace(think=(), print_raw=False)

        hri = HRI_Agent(
            "vllm",
            args,
            model=FakeBoundModel(),
            planner_agent=planner,
            memory_agent=memory,
            metrics=TurnMetrics(None),
            model_instance_id="actor-1",
        )

        self.assertIs(hri.planner_agent, planner)
        self.assertIs(hri.memory_agent, memory)
        self.assertEqual(hri.model_instance_id, "actor-1")

    def test_raw_call_observer_gets_multimodal_payload_and_is_thread_safe(self):
        observed = []
        metrics = TurnMetrics(
            None,
            raw_call_observer=lambda **record: observed.append(record),
        )
        prompt = [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AA=="},
                    },
                    {"type": "text", "text": "inspect"},
                ]
            )
        ]
        response = AIMessage(content="ok")

        threads = [
            threading.Thread(
                target=metrics.record,
                kwargs={
                    "agent": f"role-{index}",
                    "prompt_messages": prompt,
                    "response": response,
                    "elapsed_seconds": 0.1,
                },
            )
            for index in range(12)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(metrics.snapshot_calls()), 12)
        self.assertEqual(len(observed), 12)
        self.assertEqual(observed[0]["prompt_messages"][0].content[0]["type"], "image_url")

    def test_raw_call_recording_failure_fails_closed(self):
        def fail(**_record):
            raise OSError("disk full")

        metrics = TurnMetrics(None, raw_call_observer=fail)
        with self.assertRaises(MetricsRecordingError):
            metrics.record(
                agent="Planner Agent",
                prompt_messages=[HumanMessage(content="request")],
                response=AIMessage(content="response"),
                elapsed_seconds=0.1,
            )
        self.assertEqual(metrics.snapshot_calls(), ())

    def test_single_evidence_is_available_only_by_explicit_construction(self):
        transitions = []
        controller = RecedingHorizonController(
            success_confirmations=1,
            success_stability_seconds=0,
            failure_confirmations=1,
            on_transition=transitions.append,
        )

        self.assertEqual(controller._success_confirmations, 1)
        self.assertEqual(controller._failure_confirmations, 1)
        self.assertEqual(transitions, [])
        with self.assertRaises(ValueError):
            RecedingHorizonController(success_confirmations=0)

    def test_execution_fault_retains_typed_phase_and_trace(self):
        event = ExecutionEvent(
            publication_id="publication-1",
            state=ExecutionState.FAULT,
            message="failed",
            observed_at=1.0,
            error_kind=ExecutionErrorKind.GROUNDING,
            exception_type="TimeoutError",
            traceback_text="trace",
        )

        self.assertIs(event.error_kind, ExecutionErrorKind.GROUNDING)
        self.assertEqual(event.exception_type, "TimeoutError")
        with self.assertRaises(ValueError):
            ExecutionEvent(
                publication_id="publication-1",
                state=ExecutionState.SETTLED,
                message="settled",
                observed_at=1.0,
                error_kind=ExecutionErrorKind.CONTROL,
            )

    def test_cli_exposes_loopback_embedding_service(self):
        args = parse_args(
            [
                "--embedding-model",
                "embedding-test",
                "--embedding-model-base-url",
                "http://127.0.0.1:8081/v1",
            ]
        )
        self.assertEqual(args.embedding_model, "embedding-test")
        self.assertEqual(
            args.embedding_model_base_url,
            "http://127.0.0.1:8081/v1",
        )


class SimulationRecordingSeamTests(unittest.TestCase):
    def test_realtime_mujoco_evidence_pump_survives_blocking_main_work(self):
        from experiments.harness.production_engine import _RecordedFrames
        from experiments.harness.recording import ClockOrigin
        from simulation.stacking import StackingEnvironment

        origin = ClockOrigin.capture()
        recorded: list[tuple[float, str | None]] = []

        class Context:
            recorder = SimpleNamespace(origin=origin)

            @staticmethod
            def video_tick(
                third_person,
                robot_camera,
                *,
                source_elapsed_seconds,
                source_reference,
            ):
                del third_person, robot_camera
                recorded.append(
                    (
                        source_elapsed_seconds,
                        source_reference.robot_camera_id,
                    )
                )
                return len(recorded) - 1

        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            render_hz=10,
            realtime=True,
            seed=37,
        )
        self.addCleanup(environment.close)
        frames = _RecordedFrames(Context(), environment, object())
        frames.start_pump()
        self.addCleanup(frames.stop_pump, raise_error=False)

        # Model transport and graph execution are synchronous from the main
        # engine's perspective.  Evidence acquisition must remain independent.
        time.sleep(3.0)
        frames.stop_pump()

        # Three MuJoCo renders (overview, RGB, depth) are slower than the 10 Hz
        # CFR encoder on this host.  The scientific liveness invariant is that
        # fresh atomic pairs continue and any shortfall is explicitly held,
        # never that renderer throughput is relabelled as native 10 Hz.
        self.assertGreaterEqual(len(recorded), 10, recorded)
        self.assertEqual(len({item[1] for item in recorded}), len(recorded))
        gaps = [
            later[0] - earlier[0]
            for earlier, later in zip(recorded, recorded[1:])
        ]
        self.assertLessEqual(max(gaps), 0.3, gaps)

    def test_task_and_overview_views_are_synchronized_and_distinct(self):
        from simulation.stacking import StackingEnvironment

        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            render_hz=10,
            seed=31,
        )
        self.addCleanup(environment.close)

        robot, video = environment.evidence_frame_pair()
        self.assertEqual(video.sequence, robot.sequence)
        np.testing.assert_array_equal(video.task_rgb, robot.rgb)
        self.assertGreater(
            np.mean(
                np.abs(
                    video.task_rgb.astype(np.float32)
                    - video.overview_rgb.astype(np.float32)
                )
            ),
            1.0,
        )

    def test_seeded_reset_returns_exact_pre_step_hidden_state(self):
        from simulation.stacking import StackingEnvironment

        environment = StackingEnvironment(
            viewer=False,
            width=128,
            height=128,
            realtime=False,
            start=False,
            seed=7,
        )
        self.addCleanup(environment.close)

        first = environment.reset(seed=991)
        second = environment.reset(seed=991)
        for name in first.object_positions:
            np.testing.assert_array_equal(
                first.object_positions[name],
                second.object_positions[name],
            )


if __name__ == "__main__":
    unittest.main()
