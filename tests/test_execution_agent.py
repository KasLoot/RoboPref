from __future__ import annotations

import base64
import threading
import time
import unittest

import cv2
import numpy as np

from prefmem.contracts import (
    DynamicObjectScope,
    GoalContract,
    GoalProposal,
    ObservationCriterion,
    PublishedTask,
    TaskPhase,
)
from prefmem.execution.contracts import (
    AnchorKind,
    ExecutionState,
    ManipulationProgram,
    ObjectReference,
)
from prefmem.execution.frames import CameraCalibration, RGBDFrame
from prefmem.execution.gemma import GemmaExecutionCompiler
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import Sam3Client, SamDetection
from prefmem.execution.service import ExecutionService


def program_payload() -> dict:
    return {
        "schema_version": 1,
        "skill": "pick_place",
        "source": {"query": "red block", "anchor": "top_center"},
        "target": {"query": "white mat", "anchor": "surface_center"},
        "relation": "on_top",
        "orientation": "tool_down",
        "waypoints": [
            {
                "kind": "approach_source",
                "clearance_m": 0.1,
                "gripper": "open",
            },
            {"kind": "grasp_source", "clearance_m": 0.0, "gripper": "close"},
            {"kind": "lift", "clearance_m": 0.1, "gripper": "hold"},
            {
                "kind": "approach_target",
                "clearance_m": 0.1,
                "gripper": "hold",
            },
            {"kind": "place_target", "clearance_m": 0.0, "gripper": "open"},
            {"kind": "retreat", "clearance_m": 0.1, "gripper": "hold"},
        ],
    }


def task(publication_id: str = "publication-1") -> PublishedTask:
    return PublishedTask(
        plan_id="goal-1",
        revision=1,
        step_id="goal-1:r1:c1:task",
        phase=TaskPhase.STEP,
        instruction="Pick up the red block and place it on the white mat.",
        expected_observation=(
            ObservationCriterion(
                criterion_id="goal-1:r1:c1:task:c1",
                description="The red block rests at the centre of the white mat.",
            ),
        ),
        known_failure_conditions=(),
        published_at=1.0,
        publication_id=publication_id,
    )


def frame(sequence: int = 1) -> RGBDFrame:
    calibration = CameraCalibration(
        width=16,
        height=16,
        intrinsic=np.array([[10.0, 0.0, 7.5], [0.0, 10.0, 7.5], [0.0, 0.0, 1.0]]),
        world_from_camera=np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    return RGBDFrame(
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        depth_m=np.full((16, 16), 0.9, dtype=np.float32),
        calibration=calibration,
        observed_at=float(sequence),
        sequence=sequence,
    )


class FakeModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.responses.pop(0)


class MaskDetector:
    def __init__(self) -> None:
        self.block_count = 1

    def detect(self, rgb, prompt, *, threshold=None):
        if prompt == "block":
            result = []
            for index in range(self.block_count):
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[2 + index : 6 + index, 2:6] = True
                result.append(SamDetection(index, (2, 2 + index, 6, 6 + index), 16, mask))
            return tuple(result)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if prompt == "red block":
            mask[2:6, 2:6] = True
            return (SamDetection(0, (2, 2, 6, 6), 16, mask),)
        mask[9:14, 9:14] = True
        return (SamDetection(0, (9, 9, 14, 14), 25, mask),)


class SequenceRGBD:
    def __init__(self) -> None:
        self.sequence = 0

    def __call__(self):
        self.sequence += 1
        return frame(self.sequence)


class FakeController:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()
        self.holds = 0
        self.calls = []

    def execute_pick_place(self, program, source, target, cancel_event):
        self.calls.append((program, source, target))
        self.started.set()
        if self.block:
            while not self.release.wait(0.01):
                if cancel_event.is_set():
                    raise InterruptedError("scene guard cancelled motion")

    def safe_hold(self):
        self.holds += 1


class ExecutionContractTests(unittest.TestCase):
    def test_dynamic_scope_round_trips_from_proposal_to_frozen_goal(self):
        proposal = GoalProposal.from_dict(
            {
                "status": "READY",
                "goal": "Stack the blocks.",
                "final_expected_observation": ["All blocks form one stable stack."],
                "constraints": [],
                "nominal_tasks": ["Stack each block."],
                "reason": None,
                "dynamic_object_scope": {
                    "selector": "block",
                    "region": "robot_workspace",
                    "membership_rule": "PRESENT_AT_VALIDATION",
                },
            }
        )
        goal = GoalContract.from_proposal(proposal, goal_id="goal-open")
        self.assertEqual(goal.dynamic_object_scope.selector, "block")
        self.assertEqual(GoalContract.from_dict(goal.to_dict()), goal)

    def test_program_rejects_model_coordinates_and_changed_safe_sequence(self):
        payload = program_payload()
        payload["source"]["world_xyz"] = [0.5, 0.0, 0.05]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ManipulationProgram.from_dict(payload)

        payload = program_payload()
        payload["waypoints"][0], payload["waypoints"][1] = (
            payload["waypoints"][1],
            payload["waypoints"][0],
        )
        with self.assertRaisesRegex(ValueError, "exact safe six-stage"):
            ManipulationProgram.from_dict(payload)

    def test_gemma_compiler_repairs_once_and_returns_typed_program(self):
        import json

        model = FakeModel('{"skill":"unsupported"}', json.dumps(program_payload()))
        compiler = GemmaExecutionCompiler(model=model)
        program = compiler.compile(task())
        self.assertEqual(program.source.query, "red block")
        self.assertEqual(program.target.anchor, AnchorKind.SURFACE_CENTER)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("schema_correction", model.calls[1][0][1].content)

    def test_rgbd_backprojection_and_mask_grounding_are_world_framed(self):
        current = frame()
        point = current.calibration.backproject(np.array([[7.5, 7.5]]), np.array([0.9]))
        np.testing.assert_allclose(point[0], [0.5, 0.0, 0.1], atol=1e-7)
        grounded = RGBDGrounder(MaskDetector()).ground(
            current, ObjectReference("red block", AnchorKind.TOP_CENTER)
        )
        self.assertAlmostEqual(grounded.point_world[2], 0.1, places=6)
        self.assertLess(grounded.uncertainty_m, 0.01)

    def test_sam_response_includes_lossless_mask(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:5] = 255
        ok, encoded = cv2.imencode(".png", mask)
        self.assertTrue(ok)
        payload = {
            "image": {"width": 5, "height": 4},
            "count": 1,
            "detections": [
                {
                    "object_id": 0,
                    "box_xyxy": [2, 1, 5, 3],
                    "mask_area": 6,
                    "mask_png_base64": base64.b64encode(encoded).decode("ascii"),
                    "score": 0.9,
                }
            ],
        }
        detection = Sam3Client._parse_response(payload, 5, 4)[0]
        self.assertEqual(int(detection.mask.sum()), 6)
        self.assertEqual(detection.box_xyxy, (2, 1, 5, 3))


class ExecutionServiceTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("timed out waiting for execution service")

    def test_service_settles_after_compilation_grounding_and_motion(self):
        detector = MaskDetector()
        controller = FakeController()
        events = []
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
            RGBDGrounder(detector),
            controller,
            SequenceRGBD(),
            on_event=events.append,
        )
        self.addCleanup(service.stop)
        service.publish(task())
        self.wait_for(lambda: service.is_settled("publication-1"))
        self.assertEqual(len(controller.calls), 1)
        self.assertEqual(events[-1].state, ExecutionState.SETTLED)

    def test_new_dynamic_object_cancels_then_reports_scene_change(self):
        detector = MaskDetector()
        controller = FakeController(block=True)
        scene_changes = []
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
            RGBDGrounder(detector),
            controller,
            SequenceRGBD(),
            on_scene_change=scene_changes.append,
            scene_poll_interval=0.01,
            scene_change_confirmations=2,
        )
        self.addCleanup(service.stop)
        service.publish(task(), DynamicObjectScope("block"))
        self.assertTrue(controller.started.wait(1.0))
        detector.block_count = 2
        self.wait_for(lambda: bool(scene_changes))
        self.assertEqual(service.state("publication-1"), ExecutionState.CANCELLED)
        self.assertIn("began with 1", scene_changes[0].description)
        self.assertGreaterEqual(controller.holds, 1)


if __name__ == "__main__":
    unittest.main()
