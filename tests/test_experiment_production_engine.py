from __future__ import annotations

import queue
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import httpx
from langchain.messages import AIMessage
import numpy as np

from experiments.harness.production_engine import (
    ProductionDependencies,
    ProductionEngineConfig,
    ProductionMujocoEngine,
    _AttemptBinding,
    _RecordedChatBackend,
    _RecordedEmbedding,
    _RecordedFrames,
    _RecordedSamExchange,
    _SynchronizedDriver,
    _png_data_url,
    _propagate_harness_fault,
)
from experiments.harness.recording import AttemptRecorder, ClockOrigin
from experiments.harness.campaign import ScheduleCell, build_blocked_schedule
from experiments.harness.video import VideoError, source_frame_sha256
from experiments.harness.profiles import load_profiles
from experiments.harness.runner import (
    CaptureToken,
    HarnessInvalid,
    InfrastructureInterruption,
)
from experiments.harness.scenarios import load_scenarios


class FakeChatModel:
    def __init__(self, role: str, calls: list, closes: list) -> None:
        self.role = role
        self.calls = calls
        self.closes = closes
        self.bound_tools = ()

    def bind_tools(self, tools, **kwargs):
        bound = FakeChatModel(self.role, self.calls, self.closes)
        bound.bound_tools = tuple(tools)
        self.calls.append((self.role, "bind_tools", len(tools), dict(kwargs)))
        return bound

    def invoke(self, value, **kwargs):
        self.calls.append((self.role, "invoke", value, kwargs))
        return {"role": self.role, "ok": True}

    def close(self) -> None:
        self.closes.append(self.role)


class FakeEmbedding:
    def encode_document(self, values):
        return [[0.0] * 768 for _value in values]

    def encode_query(self, values):
        if isinstance(values, str):
            return [0.0] * 768
        return [[0.0] * 768 for _value in values]


class ProductionEngineTopologyTests(unittest.TestCase):
    def test_evidence_coordinator_skips_same_slot_camera_tick(self) -> None:
        origin = ClockOrigin.capture()
        base_monotonic = origin.monotonic_ns / 1_000_000_000
        elapsed_values = (134.055722104, 134.145722104, 134.245722104)
        recorded: list[str | None] = []

        class Environment:
            def __init__(self) -> None:
                self.sequence = 0

            def wait_for_frame(self, *, after_sequence, timeout):
                del timeout
                self.sequence = max(self.sequence + 1, after_sequence + 1)
                if self.sequence > len(elapsed_values):
                    raise TimeoutError("fixture exhausted")

            def evidence_frame_pair(self):
                sequence = self.sequence
                pixels = np.full((24, 32, 3), sequence, dtype=np.uint8)
                rgbd = SimpleNamespace(sequence=sequence)
                video = SimpleNamespace(
                    sequence=sequence,
                    observed_at=base_monotonic + elapsed_values[sequence - 1],
                    overview_rgb=pixels,
                    task_rgb=pixels,
                )
                return rgbd, video

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
                del third_person, robot_camera, source_elapsed_seconds
                recorded.append(source_reference.robot_camera_id)
                return len(recorded) - 1

        frames = _RecordedFrames(Context(), Environment(), object())
        frames.tick()
        frames.tick()
        self.assertEqual(recorded, ["robot_camera:1", "robot_camera:3"])

    def test_evidence_coordinator_skips_over_rate_tick_in_next_slot(self) -> None:
        origin = ClockOrigin.capture()
        base_monotonic = origin.monotonic_ns / 1_000_000_000
        elapsed_values = (15.224460614, 15.309, 15.325)
        recorded: list[str | None] = []

        class Environment:
            def __init__(self) -> None:
                self.sequence = 0

            def wait_for_frame(self, *, after_sequence, timeout):
                del timeout
                self.sequence = max(self.sequence + 1, after_sequence + 1)
                if self.sequence > len(elapsed_values):
                    raise TimeoutError("fixture exhausted")

            def evidence_frame_pair(self):
                sequence = self.sequence
                pixels = np.full((24, 32, 3), sequence, dtype=np.uint8)
                rgbd = SimpleNamespace(sequence=sequence)
                video = SimpleNamespace(
                    sequence=sequence,
                    observed_at=base_monotonic + elapsed_values[sequence - 1],
                    overview_rgb=pixels,
                    task_rgb=pixels,
                )
                return rgbd, video

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
                del third_person, robot_camera, source_elapsed_seconds
                recorded.append(source_reference.robot_camera_id)
                return len(recorded) - 1

        frames = _RecordedFrames(Context(), Environment(), object())
        frames.tick()
        frames.tick()
        self.assertEqual(recorded, ["robot_camera:1", "robot_camera:3"])

    def test_evidence_coordinator_raw_cadence_boundary_is_fail_closed(self) -> None:
        def recorded_ids(elapsed_values: tuple[float, ...]) -> list[str | None]:
            origin = ClockOrigin.capture()
            base_monotonic = origin.monotonic_ns / 1_000_000_000
            recorded: list[str | None] = []

            class Environment:
                def __init__(self) -> None:
                    self.sequence = 0

                def wait_for_frame(self, *, after_sequence, timeout):
                    del timeout
                    self.sequence = max(self.sequence + 1, after_sequence + 1)
                    if self.sequence > len(elapsed_values):
                        raise TimeoutError("fixture exhausted")

                def evidence_frame_pair(self):
                    sequence = self.sequence
                    pixels = np.full((24, 32, 3), sequence, dtype=np.uint8)
                    rgbd = SimpleNamespace(sequence=sequence)
                    video = SimpleNamespace(
                        sequence=sequence,
                        observed_at=base_monotonic + elapsed_values[sequence - 1],
                        overview_rgb=pixels,
                        task_rgb=pixels,
                    )
                    return rgbd, video

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
                    del third_person, robot_camera, source_elapsed_seconds
                    recorded.append(source_reference.robot_camera_id)
                    return len(recorded) - 1

            frames = _RecordedFrames(Context(), Environment(), object())
            frames.tick()
            frames.tick()
            return recorded

        with self.subTest("exact 90ms boundary is accepted"):
            self.assertEqual(
                recorded_ids((1.0, 1.09)),
                ["robot_camera:1", "robot_camera:2"],
            )
        with self.subTest("89ms frame is consumed without reserving its slot"):
            self.assertEqual(
                recorded_ids((1.0, 1.089, 1.1)),
                ["robot_camera:1", "robot_camera:3"],
            )

    def test_evidence_pump_records_fresh_sources_during_blocking_work(self) -> None:
        origin = ClockOrigin.capture()
        base_monotonic = origin.monotonic_ns / 1_000_000_000
        recorded: list[tuple[float, str | None]] = []
        enough = threading.Event()

        class Environment:
            def __init__(self) -> None:
                self.sequence = 0

            def wait_for_frame(self, *, after_sequence, timeout):
                del timeout
                time.sleep(0.005)
                self.sequence = max(self.sequence + 1, after_sequence + 1)

            def evidence_frame_pair(self):
                sequence = self.sequence
                observed_at = base_monotonic + sequence / 10
                pixels = np.full((24, 32, 3), 80, dtype=np.uint8)
                rgbd = SimpleNamespace(sequence=sequence)
                video = SimpleNamespace(
                    sequence=sequence,
                    observed_at=observed_at,
                    overview_rgb=pixels,
                    task_rgb=pixels,
                )
                return rgbd, video

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
                if len(recorded) >= 6:
                    enough.set()
                return len(recorded) - 1

        frames = _RecordedFrames(Context(), Environment(), object())
        frames.start_pump()
        self.assertTrue(enough.wait(1.0), recorded)
        # The main execution thread can remain blocked while the independent
        # recorder continues to acquire evidence.
        time.sleep(0.03)
        frames.stop_pump()
        self.assertGreaterEqual(len(recorded), 6)
        self.assertEqual(len({item[1] for item in recorded}), len(recorded))
        self.assertTrue(
            all(later > earlier for earlier, later in zip(recorded, recorded[1:]))
        )

    def test_video_fault_cannot_be_downgraded_to_studied_system_error(self) -> None:
        fault = VideoError("controlled evidence timing failure")
        with self.assertRaises(VideoError) as caught:
            _propagate_harness_fault(fault)
        self.assertIs(caught.exception, fault)
        self.assertIsNone(_propagate_harness_fault(RuntimeError("system failure")))

    def test_synchronized_driver_rejects_an_equal_but_distinct_clock_origin(self) -> None:
        origin = ClockOrigin.capture()

        class Recorder:
            pass

        class Driver:
            pass

        recorder = Recorder()
        recorder.origin = origin
        driver = Driver()
        driver.recorder = recorder
        proxy = _SynchronizedDriver(driver, origin=origin)
        self.assertIs(proxy._origin, origin)

        duplicate = ClockOrigin(utc=origin.utc, monotonic_ns=origin.monotonic_ns)
        with self.assertRaisesRegex(HarnessInvalid, "injected ClockOrigin"):
            _SynchronizedDriver(driver, origin=duplicate)

    def test_cell_capability_binds_primary_pilot_and_refuses_locked_fixture(self) -> None:
        scenario = load_scenarios().get("NM02")
        assignment = build_blocked_schedule(
            (
                ScheduleCell(
                    matrix_id="production_pilot",
                    scenario_id="NM02",
                    scenario_variant_id=scenario.split_variants["pilot"].variant_id,
                    profiles=("T5",),
                    repetitions=1,
                    model_backbones=("primary",),
                    executor="mujoco",
                ),
            ),
            master_seed=17,
            split="pilot",
        )[0]
        engine = ProductionMujocoEngine(
            dependencies=ProductionDependencies(
                model_client_factory=lambda *args: FakeChatModel("unused", [], []),
                embedding_client_factory=lambda *args: FakeEmbedding(),
            )
        )
        capability = engine.cell_capability(
            assignment, scenario, load_profiles().get("T5")
        )
        self.assertTrue(capability.supported, capability.reason)
        locked = replace(
            assignment,
            split="locked",
            scenario_variant_id=scenario.split_variants["locked"].variant_id,
        )
        refused = engine.cell_capability(
            locked, scenario, load_profiles().get("T5")
        )
        self.assertFalse(refused.supported)
        self.assertIn("does not enact a locked split", refused.reason)

    def test_t5_topology_constructs_five_real_role_surfaces_without_inference(self) -> None:
        calls = []
        closes = []
        factory_roles = []

        def model_factory(role, model, base_url, timeout):
            factory_roles.append((role, model, base_url, timeout))
            return FakeChatModel(role, calls, closes)

        engine = ProductionMujocoEngine(
            dependencies=ProductionDependencies(
                model_client_factory=model_factory,
                embedding_client_factory=lambda model, base_url: FakeEmbedding(),
            )
        )
        manifest = engine.topology_manifest(load_profiles().get("T5"), "mujoco")
        try:
            self.assertTrue(engine.publication_valid)
            self.assertTrue(engine.supports_locked)
            self.assertEqual(manifest["assembled_unique_agent_count"], 5)
            self.assertEqual(
                {item[0] for item in factory_roles},
                {"hri", "memory", "planner", "monitor", "validator"},
            )
            self.assertEqual(
                [item for item in calls if item[1] == "invoke"],
                [],
            )
            self.assertFalse(
                manifest["supported_execution_path"]["objective_state_visible_to_models"]
            )
            self.assertEqual(
                manifest["verification_scope"], "production_runtime_and_transport"
            )
        finally:
            engine.close()
        self.assertCountEqual(closes, ["hri", "memory", "planner", "monitor", "validator"])

    def test_unsupported_profile_executor_and_scenario_fail_closed(self) -> None:
        engine = ProductionMujocoEngine(
            dependencies=ProductionDependencies(
                model_client_factory=lambda *args: FakeChatModel("unused", [], []),
                embedding_client_factory=lambda *args: FakeEmbedding(),
            )
        )
        with self.assertRaisesRegex(HarnessInvalid, "only profile T5"):
            engine.topology_manifest(load_profiles().get("T4"), "mujoco")
        with self.assertRaisesRegex(HarnessInvalid, "only executor mujoco"):
            engine.topology_manifest(load_profiles().get("T5"), "synthetic_event")
        unsupported = next(
            scenario
            for scenario in load_scenarios().scenarios
            if scenario.scenario_id != "NM02"
        )
        with self.assertRaisesRegex(HarnessInvalid, "only scenario NM02"):
            engine.trigger_handlers(unsupported)

    def test_assembled_production_planner_uses_recorded_exact_frame(self) -> None:
        calls = []
        closes = []

        class PlannerModel(FakeChatModel):
            def invoke(self, value, **kwargs):
                self.calls.append((self.role, "invoke", value, kwargs))
                return AIMessage(
                    content=(
                        '{"status":"READY","goal":"Put the red cube on the '
                        'target pad.","final_expected_observation":["The red cube '
                        'rests on the target pad."],"constraints":[],"nominal_tasks":'
                        '["Put the red cube on the target pad."],"reason":"The '
                        'requested pick and place is feasible."}'
                    )
                )

        def model_factory(role, model, base_url, timeout):
            del model, base_url, timeout
            cls = PlannerModel if role == "planner" else FakeChatModel
            return cls(role, calls, closes)

        engine = ProductionMujocoEngine(
            dependencies=ProductionDependencies(
                model_client_factory=model_factory,
                embedding_client_factory=lambda model, base_url: FakeEmbedding(),
            )
        )
        engine.topology_manifest(load_profiles().get("T5"), "mujoco")
        pending = engine._pending
        assert pending is not None
        attempt = FakeAttemptContext()
        pending.binding.bind_context(attempt)
        token = capture_token()
        pending.binding.set_capture(token)
        planner = pending.assembled.components["planner"]

        proposal = planner.preview(
            "Put the red cube on the target pad.",
            {
                "type": "image_url",
                "image_url": {"url": _png_data_url(token)},
            },
        )

        self.assertEqual(proposal.status.value, "READY")
        self.assertEqual(proposal.goal, "Put the red cube on the target pad.")
        self.assertEqual(len(attempt.exchanges), 1)
        self.assertEqual(attempt.exchanges[0]["logical_agent"], "planner")
        engine.close()

    def test_service_urls_must_be_loopback(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            ProductionEngineConfig(upper_model_base_url="https://models.example/v1")

    def test_nm02_trigger_delivers_input_without_executing_behavior_in_handler(self) -> None:
        class Driver:
            def __init__(self) -> None:
                self.responses = []

            def record_trigger_response(self, trigger_id, *, event_kind, **payload):
                self.responses.append((trigger_id, event_kind, payload))

        engine = ProductionMujocoEngine()
        scenario = load_scenarios().get("NM02")
        inbox = queue.SimpleQueue()
        engine._active_messages = inbox
        driver = Driver()
        trigger = scenario.triggers[0]
        engine.trigger_handlers(scenario)[trigger.trigger_id](trigger, driver)

        self.assertEqual(inbox.get_nowait(), scenario.user_script[0])
        self.assertEqual(len(driver.responses), 1)
        self.assertEqual(driver.responses[0][1], "scripted_input_delivered")
        self.assertEqual(driver.responses[0][2]["boundary"], "before_goal_proposal")


class FakeDriver:
    def __init__(self) -> None:
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))
        return {"event_id": f"E{len(self.events):06d}"}

    def dialogue_act(self, kind, **payload):
        return self.emit(f"dialogue:{kind}", **payload)


class FakeAttemptContext:
    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.exchanges = []
        self.exchange_kinds = []

    def invoke_model_exchange(self, **kwargs):
        self.exchange_kinds.append("visual")
        self.exchanges.append(kwargs)
        return kwargs["invoke"](kwargs["request"])

    def invoke_nonvisual_model_exchange(self, **kwargs):
        self.exchange_kinds.append("nonvisual")
        self.exchanges.append(kwargs)
        return kwargs["invoke"](kwargs["request"])


def capture_token() -> CaptureToken:
    return CaptureToken(
        frame_sequence=7,
        video_frame_index=3,
        camera_frame_id="robot_camera:7",
        robot_png=b"exact-png",
        source_frame_sha256="a" * 64,
        capture_monotonic_ns=10,
        capture_elapsed_seconds=0.1,
    )


class RecordedBackendTests(unittest.TestCase):
    def test_sam_exchange_records_exact_jpeg_and_lossless_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-sam",
                attempt_number=1,
            )

            class Attempt(FakeAttemptContext):
                def __init__(self) -> None:
                    super().__init__()
                    self.recorder = recorder

            source_bgr = np.zeros((12, 16, 3), dtype=np.uint8)
            source_bgr[:, :, 1] = 180
            ok, png = cv2.imencode(".png", source_bgr)
            self.assertTrue(ok)
            ok, jpeg = cv2.imencode(
                ".jpg",
                source_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
            self.assertTrue(ok)
            binding = _AttemptBinding(lambda service, url: {"ok": True})
            binding.bind_context(Attempt())
            binding.set_capture(
                CaptureToken(
                    frame_sequence=9,
                    video_frame_index=4,
                    camera_frame_id="robot_camera:9",
                    robot_png=png.tobytes(),
                    source_frame_sha256=source_frame_sha256(source_bgr),
                    capture_monotonic_ns=recorder.origin.monotonic_ns + 100_000_000,
                    capture_elapsed_seconds=0.1,
                )
            )
            exchange = _RecordedSamExchange(binding, "http://127.0.0.1:9000")
            token = exchange.start(
                prompt="red cube",
                threshold=0.5,
                image_jpeg=jpeg.tobytes(),
            )
            raw_response = b'{"image":{"width":16,"height":12},"count":0,"detections":[]}'
            exchange.end(token, status_code=200, response_body=raw_response)
            failed_token = exchange.start(
                prompt="blue cube",
                threshold=0.4,
                image_jpeg=jpeg.tobytes(),
            )
            exchange.error(
                failed_token,
                error=httpx.ConnectError("controlled detector disconnect"),
                status_code=None,
                response_body=None,
            )

            frame_records = [
                json.loads(line)
                for line in (recorder.attempt_dir / "frame_requests.jsonl")
                .read_text()
                .splitlines()
            ]
            request = frame_records[0]
            self.assertTrue(request["image_path"].endswith(".jpg"))
            self.assertTrue(request["source_image_path"].endswith(".png"))
            self.assertEqual(request["camera_frame_id"], "robot_camera:9")
            self.assertEqual(request["response_id"], None)
            self.assertEqual(frame_records[1]["response_id"], "RESP-000001")
            self.assertEqual(frame_records[2]["request_id"], "REQ-000002")
            self.assertEqual(frame_records[3]["record_type"], "frame_error_link")
            self.assertEqual(frame_records[3]["request_id"], "REQ-000002")
            self.assertEqual(frame_records[3]["call_id"], "CALL-000002")
            call_records = [
                json.loads(line)
                for line in (recorder.attempt_dir / "model_calls.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(call_records[-1]["record_type"], "model_call_error")
            self.assertIsNone(call_records[-1]["response_id"])
            self.assertEqual(
                frame_records[3]["event_ids"], [call_records[-1]["event_id"]]
            )
            self.assertEqual(recorder.model_calls.active_call_ids, ())

    def test_embedding_call_records_prefixed_request_and_exact_response(self) -> None:
        class Calls:
            def __init__(self) -> None:
                self.records = []

            def start(self, **value):
                self.records.append(("start", value))

            def end(self, **value):
                self.records.append(("end", value))

            def error(self, **value):
                self.records.append(("error", value))

        class Recorder:
            def __init__(self) -> None:
                self.model_calls = Calls()

        class Attempt(FakeAttemptContext):
            def __init__(self) -> None:
                super().__init__()
                self.recorder = Recorder()

        class Embedding(FakeEmbedding):
            DOCUMENT_PROMPT = "document: "
            QUERY_PROMPT = "query: "

        attempt = Attempt()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        backend = _RecordedEmbedding(
            target=Embedding(),
            binding=binding,
            model_id="embedding-id",
            base_url="http://127.0.0.1:8080/v1",
        )

        response = backend.encode_query("red cube")

        self.assertEqual(response, [0.0] * 768)
        self.assertEqual([kind for kind, _ in attempt.recorder.model_calls.records], ["start", "end"])
        started = attempt.recorder.model_calls.records[0][1]
        ended = attempt.recorder.model_calls.records[1][1]
        self.assertEqual(started["logical_agent"], "memory_embedding")
        self.assertEqual(started["model_id"], "embedding-id")
        self.assertEqual(started["request"]["input"], ["query: red cube"])
        self.assertEqual(ended["response"], [0.0] * 768)
        self.assertEqual(
            [payload["logical_agent"] for kind, payload in attempt.driver.events if kind.startswith("model_")],
            ["memory_embedding", "memory_embedding"],
        )

    def test_visual_call_requires_and_records_exact_pixels_before_transport(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        token = capture_token()
        binding.set_capture(token)
        calls = []
        target = FakeChatModel("planner", calls, [])
        backend = _RecordedChatBackend(
            target=target,
            binding=binding,
            logical_role="planner",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-planner-openai",
            endpoint_label="upper-model-loopback",
        )
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _png_data_url(token)}},
                        {"type": "text", "text": "plan"},
                    ],
                }
            ]
        }
        response = backend.invoke(request, temperature=0)

        self.assertEqual(response, {"role": "planner", "ok": True})
        self.assertEqual(len(attempt.exchanges), 1)
        self.assertIs(attempt.exchanges[0]["capture"], token)
        self.assertEqual(len([item for item in calls if item[1] == "invoke"]), 1)

        with self.assertRaisesRegex(HarnessInvalid, "exact recorded frame"):
            backend.invoke({"messages": [{"role": "user", "content": "no image"}]})
        with self.assertRaisesRegex(HarnessInvalid, "exact recorded frame"):
            backend.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _png_data_url(token),
                                }
                            ],
                        }
                    ],
                    "metadata": {"caption": _png_data_url(token)},
                }
            )

    def test_bound_tool_contract_is_persisted_in_semantic_request(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        token = capture_token()
        binding.set_capture(token)
        calls = []
        backend = _RecordedChatBackend(
            target=FakeChatModel("hri", calls, []),
            binding=binding,
            logical_role="hri",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-hri-openai",
            endpoint_label="upper-model-loopback",
        )
        tool = {
            "type": "function",
            "function": {
                "name": "request_goal_preview",
                "description": "Stage a goal preview.",
                "parameters": {
                    "type": "object",
                    "properties": {"clarified_goal": {"type": "string"}},
                    "required": ["clarified_goal"],
                },
            },
        }
        bound = backend.bind_tools(
            [tool],
            tool_choice="request_goal_preview",
            parallel_tool_calls=False,
        )

        bound.invoke({"image": _png_data_url(token), "turn": "preview"})

        request = attempt.exchanges[0]["request"]
        contract = request["bound_tool_contracts"][0]
        self.assertEqual(contract["serialized_tool_definitions"], [tool])
        self.assertEqual(
            contract["bind_options"],
            {
                "tool_choice": "request_goal_preview",
                "parallel_tool_calls": False,
            },
        )
        self.assertEqual(contract["bound_target_kwargs"]["tools"], [tool])
        self.assertEqual(attempt.exchange_kinds, ["visual"])

    def test_text_only_executor_records_nonvisual_exchange_without_capture(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        calls = []
        backend = _RecordedChatBackend(
            target=FakeChatModel("executor", calls, []),
            binding=binding,
            logical_role="executor",
            model_id="executor-model-id",
            base_url="http://127.0.0.1:8001/v1",
            service_label="execution-compiler-openai",
            endpoint_label="execution-model-loopback",
        )

        response = backend.invoke(
            {"messages": [{"role": "user", "content": "compile this task"}]}
        )

        self.assertEqual(response, {"role": "executor", "ok": True})
        self.assertEqual(attempt.exchange_kinds, ["nonvisual"])
        self.assertNotIn("capture", attempt.exchanges[0])
        self.assertEqual(
            attempt.exchanges[0]["request"]["messages"]["messages"][0]["content"],
            "compile this task",
        )
        self.assertEqual(len([item for item in calls if item[1] == "invoke"]), 1)

    def test_nested_visual_calls_resolve_the_capture_embedded_in_each_request(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        frame_a = capture_token()
        frame_b = replace(
            frame_a,
            frame_sequence=8,
            video_frame_index=4,
            camera_frame_id="robot_camera:8",
            source_frame_sha256="b" * 64,
            capture_monotonic_ns=20,
            capture_elapsed_seconds=0.2,
        )
        calls = []
        hri = _RecordedChatBackend(
            target=FakeChatModel("hri", calls, []),
            binding=binding,
            logical_role="hri",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-hri-openai",
            endpoint_label="upper-model-loopback",
        )
        planner = _RecordedChatBackend(
            target=FakeChatModel("planner", calls, []),
            binding=binding,
            logical_role="planner",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-planner-openai",
            endpoint_label="upper-model-loopback",
        )

        binding.set_capture(frame_a)
        with binding.pin_capture("hri", frame_a):
            hri.invoke({"image": _png_data_url(frame_a), "turn": "first"})
            binding.set_capture(frame_b)
            planner.invoke({"image": _png_data_url(frame_b), "turn": "nested"})
            # A and B have identical PNG bytes.  Planner uses its matching
            # thread-local B provenance, while HRI retains its scoped A pin.
            hri.invoke({"image": _png_data_url(frame_a), "turn": "second"})

        self.assertEqual(len(attempt.exchanges), 3)
        self.assertIs(attempt.exchanges[0]["capture"], frame_a)
        self.assertIs(attempt.exchanges[1]["capture"], frame_b)
        self.assertIs(attempt.exchanges[2]["capture"], frame_a)
        self.assertEqual(
            attempt.exchanges[2]["request"]["messages"]["image"],
            _png_data_url(frame_a),
        )

    def test_hri_turn_pins_its_capture_across_nested_graph_acquisitions(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        frame_a = capture_token()
        frame_b = replace(
            frame_a,
            frame_sequence=8,
            video_frame_index=4,
            camera_frame_id="robot_camera:8",
            source_frame_sha256="b" * 64,
            capture_monotonic_ns=20,
            capture_elapsed_seconds=0.2,
        )
        hri_backend = _RecordedChatBackend(
            target=FakeChatModel("hri", [], []),
            binding=binding,
            logical_role="hri",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-hri-openai",
            endpoint_label="upper-model-loopback",
        )

        class Frames:
            def capture_model_frame(self):
                binding.set_capture(frame_a)
                return (
                    type(
                        "Frame",
                        (),
                        {
                            "sequence": frame_a.frame_sequence,
                            "image_block": {
                                "type": "image_url",
                                "image_url": {"url": _png_data_url(frame_a)},
                            },
                        },
                    )(),
                    frame_a,
                )

        class HriGraph:
            def invoke(self, value, **kwargs):
                del value, kwargs
                binding.set_capture(frame_b)
                hri_backend.invoke({"image": _png_data_url(frame_a)})
                return {"messages": []}

        pending = type(
            "Pending",
            (),
            {"binding": binding, "conversation_id": "conversation-test"},
        )()
        hri = type("Hri", (), {"hri_agent": HriGraph()})()

        ProductionMujocoEngine()._invoke_hri_turn(
            hri=hri,
            frames=Frames(),
            pending=pending,
            message="move the red cube",
            script_index=0,
        )

        self.assertEqual(len(attempt.exchanges), 1)
        self.assertIs(attempt.exchanges[0]["capture"], frame_a)
        self.assertEqual(binding._pinned_captures, {})

    def test_visual_capture_resolution_rejects_unregistered_tampered_and_ambiguous_images(self) -> None:
        attempt = FakeAttemptContext()
        binding = _AttemptBinding(lambda service, url: {"ok": True})
        binding.bind_context(attempt)
        frame_a = capture_token()
        frame_b = replace(
            frame_a,
            frame_sequence=8,
            video_frame_index=4,
            camera_frame_id="robot_camera:8",
            robot_png=b"second-png",
            source_frame_sha256="b" * 64,
        )
        binding.set_capture(frame_a)
        backend = _RecordedChatBackend(
            target=FakeChatModel("planner", [], []),
            binding=binding,
            logical_role="planner",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-planner-openai",
            endpoint_label="upper-model-loopback",
        )

        unregistered_url = _png_data_url(frame_b)
        with self.assertRaisesRegex(HarnessInvalid, "unregistered"):
            backend.invoke({"image": unregistered_url})
        with self.assertRaisesRegex(HarnessInvalid, "tampered or unregistered"):
            backend.invoke({"image": _png_data_url(frame_a) + "tampered"})

        binding.set_capture(frame_b)
        with binding.pin_capture("planner", frame_a):
            with self.assertRaisesRegex(HarnessInvalid, "role-scoped"):
                backend.invoke({"image": _png_data_url(frame_b)})
        with self.assertRaisesRegex(HarnessInvalid, "ambiguous"):
            backend.invoke(
                {"images": [_png_data_url(frame_a), _png_data_url(frame_b)]}
            )

        same_pixels_new_identity = replace(
            frame_a,
            frame_sequence=9,
            video_frame_index=5,
            camera_frame_id="robot_camera:9",
            capture_monotonic_ns=30,
            capture_elapsed_seconds=0.3,
        )
        binding.set_capture(same_pixels_new_identity)
        outcome = queue.SimpleQueue()

        def invoke_without_scope() -> None:
            try:
                backend.invoke({"image": _png_data_url(frame_a)})
            except BaseException as error:
                outcome.put(error)

        worker = threading.Thread(target=invoke_without_scope)
        worker.start()
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        error = outcome.get_nowait()
        self.assertIsInstance(error, HarnessInvalid)
        self.assertIn("ambiguous", str(error))

        self.assertEqual(attempt.exchanges, [])

    def test_failed_transport_plus_failed_probe_latches_typed_infrastructure(self) -> None:
        class FailingModel:
            def invoke(self, value, **kwargs):
                del value, kwargs
                request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat")
                raise httpx.ConnectError("disconnected", request=request)

        attempt = FakeAttemptContext()
        binding = _AttemptBinding(
            lambda service, url: {
                "ok": False,
                "probe": "GET /models",
                "error_type": "ConnectError",
            }
        )
        binding.bind_context(attempt)
        token = capture_token()
        binding.set_capture(token)
        stops = []
        binding.bind_safe_stop(lambda: stops.append("safe") is None or True)
        backend = _RecordedChatBackend(
            target=FailingModel(),
            binding=binding,
            logical_role="planner",
            model_id="model-id",
            base_url="http://127.0.0.1:8000/v1",
            service_label="upper-planner-openai",
            endpoint_label="upper-model-loopback",
        )
        request = {
            "image": _png_data_url(token),
            "messages": [{"role": "user", "content": "plan"}],
        }
        with self.assertRaises(InfrastructureInterruption) as caught:
            backend.invoke(request)

        self.assertEqual(stops, ["safe"])
        self.assertIs(binding.infrastructure_failure, caught.exception)
        self.assertTrue(caught.exception.evidence.safe_stop_confirmed)
        self.assertEqual(
            caught.exception.evidence.health_observations[0]["ok"], False
        )

    def test_nonvisual_transport_failure_preserves_infrastructure_classification(self) -> None:
        class FailingModel:
            def invoke(self, value, **kwargs):
                del value, kwargs
                request = httpx.Request("POST", "http://127.0.0.1:8001/v1/chat")
                raise httpx.ConnectError("disconnected", request=request)

        attempt = FakeAttemptContext()
        binding = _AttemptBinding(
            lambda service, url: {
                "ok": False,
                "probe": "GET /models",
                "error_type": "ConnectError",
            }
        )
        binding.bind_context(attempt)
        stops = []
        binding.bind_safe_stop(lambda: stops.append("safe") is None or True)
        backend = _RecordedChatBackend(
            target=FailingModel(),
            binding=binding,
            logical_role="executor",
            model_id="executor-model-id",
            base_url="http://127.0.0.1:8001/v1",
            service_label="execution-compiler-openai",
            endpoint_label="execution-model-loopback",
        )

        with self.assertRaises(InfrastructureInterruption):
            backend.invoke({"messages": [{"role": "user", "content": "compile"}]})

        self.assertEqual(attempt.exchange_kinds, ["nonvisual"])
        self.assertEqual(stops, ["safe"])


if __name__ == "__main__":
    unittest.main()
