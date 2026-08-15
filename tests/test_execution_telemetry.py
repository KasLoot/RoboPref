from __future__ import annotations

import json
import time
import unittest

from prefmem.execution.contracts import ManipulationProgram
from prefmem.execution.gemma import GemmaExecutionCompiler
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.service import ExecutionService
from tests.test_execution_agent import (
    FakeController,
    FakeModel,
    MaskDetector,
    SequenceRGBD,
    program_payload,
    task,
)


class ExecutionTelemetryTests(unittest.TestCase):
    def wait_for(self, predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("timed out waiting for execution telemetry")

    def test_service_emits_compiler_frame_grounding_and_perception_records(self) -> None:
        traces: list[dict] = []
        perception: list[dict] = []
        service = ExecutionService(
            type(
                "Compiler",
                (),
                {
                    "compile": lambda _self, _task: ManipulationProgram.from_dict(
                        program_payload()
                    )
                },
            )(),
            RGBDGrounder(MaskDetector()),
            FakeController(),
            SequenceRGBD(),
            on_trace=traces.append,
            on_perception=perception.append,
        )
        self.addCleanup(service.stop)

        service.publish(task())
        self.wait_for(lambda: service.is_settled("publication-1"))

        kinds = [event["kind"] for event in traces]
        self.assertIn("COMPILER_OUTPUT_ACCEPTED", kinds)
        self.assertIn("RGBD_FRAME_CAPTURED", kinds)
        self.assertEqual(kinds.count("GROUNDING_COMPLETE"), 2)
        grounding = next(
            event for event in traces if event["kind"] == "GROUNDING_COMPLETE"
        )
        self.assertGreater(grounding["diagnostics"]["raw_mask_points"], 0)
        self.assertGreater(grounding["diagnostics"]["retained_depth_points"], 0)
        self.assertEqual(
            [item["kind"] for item in perception],
            ["RGBD_FRAME", "GROUNDING_MASK", "GROUNDING_MASK"],
        )

    def test_execution_compiler_records_each_schema_attempt(self) -> None:
        traces: list[dict] = []
        model = FakeModel(
            '{"skill":"unsupported"}',
            json.dumps(program_payload()),
        )
        compiler = GemmaExecutionCompiler(
            model=model,
            trace_callback=traces.append,
        )

        compiler.compile(task())

        self.assertEqual(len(traces), 2)
        self.assertIsNotNone(traces[0]["validation_error"])
        self.assertIsNone(traces[1]["validation_error"])
        self.assertEqual(traces[1]["parsed_program"]["skill"], "pick_place")


if __name__ == "__main__":
    unittest.main()
