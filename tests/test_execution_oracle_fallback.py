from __future__ import annotations

import time
import unittest

import numpy as np

from prefmem.execution.contracts import ManipulationProgram
from prefmem.execution.fallback import (
    OracleGrounding,
    SIMULATOR_GROUND_TRUTH_SOURCE,
)
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import SamServiceError
from prefmem.execution.service import ExecutionService
from tests.test_execution_agent import (
    FakeController,
    MaskDetector,
    SequenceRGBD,
    program_payload,
    task,
)


class EmptySourceDetector(MaskDetector):
    def detect(self, rgb, prompt, *, threshold=None):
        if prompt == "red block":
            return ()
        return super().detect(rgb, prompt, threshold=threshold)


class ServiceFailureDetector(MaskDetector):
    def detect(self, rgb, prompt, *, threshold=None):
        raise SamServiceError("forwarded SAM endpoint disconnected")


class RecordingOracleProvider:
    source_id = SIMULATOR_GROUND_TRUTH_SOURCE

    def __init__(self) -> None:
        self.calls = []

    def ground_from_simulator_truth(
        self,
        frame,
        reference,
        *,
        role,
        strict_failure_reason_code,
    ):
        self.calls.append(
            (reference.query, role, strict_failure_reason_code, frame.sequence)
        )
        return OracleGrounding(
            query=reference.query,
            anchor=reference.anchor,
            point_world=np.array([0.42, -0.08, 0.052]),
            frame_sequence=frame.sequence,
            diagnostics={"fixture": "explicit-test-simulator-truth"},
        )


def compiler():
    return type(
        "Compiler",
        (),
        {
            "compile": lambda _self, _task: ManipulationProgram.from_dict(
                program_payload()
            )
        },
    )()


class ExecutionOracleFallbackTests(unittest.TestCase):
    def wait_for(self, predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("timed out waiting for execution service")

    def test_fallback_configuration_requires_explicit_flag_and_provider(self) -> None:
        provider = RecordingOracleProvider()
        common = (
            compiler(),
            RGBDGrounder(MaskDetector()),
            FakeController(),
            SequenceRGBD(),
        )
        with self.assertRaisesRegex(ValueError, "both the explicit opt-in"):
            ExecutionService(*common, oracle_grounding_provider=provider)
        with self.assertRaisesRegex(ValueError, "both the explicit opt-in"):
            ExecutionService(*common, enable_oracle_grounding_fallback=True)

    def test_zero_detection_is_strict_failure_but_assisted_motion_continues(self) -> None:
        provider = RecordingOracleProvider()
        controller = FakeController()
        traces: list[dict] = []
        perception: list[dict] = []
        service = ExecutionService(
            compiler(),
            RGBDGrounder(EmptySourceDetector()),
            controller,
            SequenceRGBD(),
            enable_oracle_grounding_fallback=True,
            oracle_grounding_provider=provider,
            on_trace=traces.append,
            on_perception=perception.append,
        )
        self.addCleanup(service.stop)

        service.publish(task())
        self.wait_for(lambda: service.is_settled("publication-1"))

        self.assertEqual(
            provider.calls,
            [("red block", "source", "SAM_ZERO_DETECTIONS", 1)],
        )
        self.assertEqual(len(controller.calls), 1)
        np.testing.assert_allclose(
            controller.calls[0][1],
            [0.42, -0.08, 0.052],
        )
        outcome = service.outcome("publication-1")
        assert outcome is not None
        self.assertEqual(outcome["strict_system_result"], "FAIL_GROUNDING")
        self.assertTrue(outcome["oracle_fallback_used"])
        self.assertEqual(
            outcome["oracle_fallback_source"],
            SIMULATOR_GROUND_TRUTH_SOURCE,
        )
        self.assertEqual(outcome["assisted_continuation_result"], "PASS")
        self.assertEqual(outcome["downstream_execution_result"], "PASS")
        self.assertEqual(
            [item["strict_grounding_result"] for item in outcome["grounding_attempts"]],
            ["FAIL", "PASS"],
        )
        kinds = [event["kind"] for event in traces]
        self.assertIn("SAM_GROUNDING_FAILED", kinds)
        self.assertIn("ORACLE_GROUNDING_FALLBACK_USED", kinds)
        self.assertEqual(
            [item["kind"] for item in perception],
            [
                "RGBD_FRAME",
                "GROUNDING_FAILURE",
                "ORACLE_GROUNDING_FALLBACK",
                "GROUNDING_MASK",
            ],
        )

    def test_default_path_records_detection_failure_without_assistance(self) -> None:
        controller = FakeController()
        service = ExecutionService(
            compiler(),
            RGBDGrounder(EmptySourceDetector()),
            controller,
            SequenceRGBD(),
        )
        self.addCleanup(service.stop)

        service.publish(task())
        self.wait_for(
            lambda: getattr(service.state("publication-1"), "value", None)
            == "FAULT"
        )

        self.assertEqual(controller.calls, [])
        outcome = service.outcome("publication-1")
        assert outcome is not None
        self.assertEqual(outcome["strict_system_result"], "FAIL_GROUNDING")
        self.assertFalse(outcome["oracle_fallback_used"])
        self.assertEqual(
            outcome["assisted_continuation_result"],
            "NOT_APPLICABLE",
        )

    def test_successful_sam_grounding_never_consults_oracle(self) -> None:
        provider = RecordingOracleProvider()
        service = ExecutionService(
            compiler(),
            RGBDGrounder(MaskDetector()),
            FakeController(),
            SequenceRGBD(),
            enable_oracle_grounding_fallback=True,
            oracle_grounding_provider=provider,
        )
        self.addCleanup(service.stop)

        service.publish(task())
        self.wait_for(lambda: service.is_settled("publication-1"))

        self.assertEqual(provider.calls, [])
        outcome = service.outcome("publication-1")
        assert outcome is not None
        self.assertEqual(outcome["strict_system_result"], "PASS")
        self.assertFalse(outcome["oracle_fallback_used"])

    def test_service_outage_is_not_oracle_eligible(self) -> None:
        provider = RecordingOracleProvider()
        traces: list[dict] = []
        service = ExecutionService(
            compiler(),
            RGBDGrounder(ServiceFailureDetector()),
            FakeController(),
            SequenceRGBD(),
            enable_oracle_grounding_fallback=True,
            oracle_grounding_provider=provider,
            on_trace=traces.append,
        )
        self.addCleanup(service.stop)

        service.publish(task())
        self.wait_for(
            lambda: getattr(service.state("publication-1"), "value", None)
            == "FAULT"
        )

        self.assertEqual(provider.calls, [])
        outcome = service.outcome("publication-1")
        assert outcome is not None
        self.assertEqual(outcome["strict_system_result"], "FAIL_EXECUTION")
        self.assertFalse(outcome["oracle_fallback_used"])
        service_failure = next(
            item
            for item in traces
            if item["kind"] == "SAM_GROUNDING_SERVICE_FAILURE"
        )
        self.assertFalse(service_failure["oracle_fallback_eligible"])


if __name__ == "__main__":
    unittest.main()
