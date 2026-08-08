from __future__ import annotations

import contextlib
import io
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace

from prefmem.agents.hri import HRI_Agent


class HRIValidationReportTests(unittest.TestCase):
    def test_ordinary_string_notification_is_unchanged(self) -> None:
        notification = "Planner cycle failed: connection timed out"

        self.assertEqual(
            HRI_Agent._render_runtime_notification(notification),
            notification,
        )

    def test_complete_report_has_summary_and_broad_checklist_only(self) -> None:
        report = {
            "validation_id": "validation-4",
            "status": "COMPLETE",
            "summary": "Task complete — all required outcomes are visibly met.",
            "checklist": [
                {
                    "label": "Trash items are in the bin.",
                    "state": "MET",
                },
                {
                    "label": "Electronics are in the box.",
                    "state": "MET",
                },
            ],
            "evidence_requests": [],
            "observation": "The floor is clear.",
            "detailed_checklist": [
                {
                    "description": "Battery 17 is behind the front box wall.",
                    "status": "MET",
                }
            ],
        }

        rendered = HRI_Agent._render_runtime_notification(report)

        self.assertIn(
            "Task complete — all required outcomes are visibly met.",
            rendered,
        )
        self.assertIn("✓ Trash items are in the bin.", rendered)
        self.assertIn("✓ Electronics are in the box.", rendered)
        self.assertNotIn("Battery 17", rendered)
        self.assertNotIn("The floor is clear", rendered)

    def test_incomplete_wrapped_object_includes_supplied_next_action(self) -> None:
        report = SimpleNamespace(
            status=SimpleNamespace(value="INCOMPLETE"),
            brief_checklist=(
                SimpleNamespace(
                    description="Trash items are in the bin.",
                    status=SimpleNamespace(value="MET"),
                ),
                SimpleNamespace(
                    description="Electronics are in the box.",
                    status=SimpleNamespace(value="NOT_MET"),
                    brief_evidence="One battery remains on the floor.",
                ),
            ),
            next_action="Replan the remaining electronics correction.",
        )
        notification = {"validation_report": report}

        rendered = HRI_Agent._render_runtime_notification(notification)

        self.assertIn(
            "Task incomplete — 1 of 2 required outcomes were verified.",
            rendered,
        )
        self.assertIn("✗ Electronics are in the box.", rendered)
        self.assertIn("One battery remains on the floor.", rendered)
        self.assertIn(
            "Next action: Replan the remaining electronics correction.",
            rendered,
        )

    def test_needs_evidence_report_lists_requests(self) -> None:
        rendered = HRI_Agent._render_runtime_notification(
            {
                "notification_type": "VALIDATION_REPORT",
                "validation_status": "NEEDS_EVIDENCE",
                "checklist": [
                    {
                        "requirement": "All electronics are in the box.",
                        "result": "UNKNOWN",
                        "observation": "The box contents are occluded.",
                    }
                ],
                "evidence_requests": [
                    "Point the camera inside the box.",
                    "Hold the camera steady.",
                ],
            }
        )

        self.assertIn(
            "Validation needs more evidence — 1 of 1 required outcomes "
            "could not be verified.",
            rendered,
        )
        self.assertIn("? All electronics are in the box.", rendered)
        self.assertIn("Evidence needed:\n- Point the camera", rendered)
        self.assertIn("- Hold the camera steady.", rendered)

    def test_notification_queue_uses_structured_renderer(self) -> None:
        hri = object.__new__(HRI_Agent)
        notifications = queue.SimpleQueue()
        notifications.put(
            {
                "type": "FINAL_VALIDATION_REPORT",
                "status": "INCOMPLETE",
                "broad_checklist": [
                    {"item": "The red block is in the tray.", "status": "NOT_MET"}
                ],
            }
        )
        hri.runtime = SimpleNamespace(notifications=notifications)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hri._print_runtime_notifications()

        rendered = output.getvalue()
        self.assertIn("PrefMem: Task incomplete", rendered)
        self.assertIn("✗ The red block is in the tray.", rendered)

    def test_console_never_claims_complete_before_controller(self) -> None:
        hri = object.__new__(HRI_Agent)
        notifications = queue.SimpleQueue()
        notifications.put(
            {
                "status": "COMPLETE",
                "summary": "Task complete — all required outcomes are met.",
                "checklist": [
                    {"label": "The red block is in the tray.", "state": "MET"}
                ],
                "evidence_requests": [],
            }
        )
        hri.runtime = SimpleNamespace(
            notifications=notifications,
            context_dict=lambda: {"state": "FINAL_VALIDATION"},
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hri._print_runtime_notifications()

        rendered = output.getvalue()
        self.assertNotIn("Task complete", rendered)
        self.assertIn("has not marked the task complete", rendered)
        self.assertIn("controller state: FINAL_VALIDATION", rendered)
        self.assertIn("✓ The red block is in the tray.", rendered)

    def test_prompt_keeps_completion_authority_with_controller(self) -> None:
        prompt = Path(
            "src/prefmem/agents/prompt/hri/hri-prompt-v5.md"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn("detailed checklist remains internal", prompt)
        self.assertIn("broad checklist is", prompt)
        self.assertIn("authoritative runtime", prompt)
        self.assertIn("only when the authoritative runtime", prompt)

    def test_summary_monitor_events_are_live_and_hide_inference_noise(self) -> None:
        hri = object.__new__(HRI_Agent)
        monitor_events = queue.Queue()
        monitor_events.put(
            {
                "kind": "MONITOR_EVENT",
                "event": "INFERENCE_STARTED",
                "publication_id": "goal:r1:c1:a1",
                "frame_sequence": 7,
            }
        )
        monitor_events.put(
            {
                "kind": "MONITOR_EVENT",
                "event": "ASSESSMENT",
                "publication_id": "goal:r1:c1:a1",
                "step_id": "place-green",
                "frame_sequence": 8,
                "task_status": "SUCCESS",
                "disposition": "ACCEPTED",
                "executor_state": "SETTLED",
                "criteria": [
                    {
                        "id": "green-on-red",
                        "description": "The green block is on the red block.",
                        "state": "MET",
                    }
                ],
                "observation": "The green block is visibly resting on the red block.",
                "success_confirmation": {"count": 1, "required": 2},
            }
        )
        hri.args = SimpleNamespace(monitor_events="summary")
        hri.runtime = SimpleNamespace(
            notifications=queue.SimpleQueue(),
            monitor_events=monitor_events,
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hri._print_runtime_notifications()

        rendered = output.getvalue()
        self.assertNotIn("inference started", rendered)
        self.assertIn("[Monitor] assessment", rendered)
        self.assertIn("status=SUCCESS", rendered)
        self.assertIn("confirmation=1/2", rendered)
        self.assertIn("MET: The green block is on the red block.", rendered)

    def test_verbose_monitor_event_includes_inference_latency(self) -> None:
        rendered = HRI_Agent._render_monitor_event(
            {
                "event": "INFERENCE_COMPLETED",
                "publication_id": "goal:r1:c1:a1",
                "frame_sequence": 11,
                "elapsed_seconds": 0.625,
            }
        )

        self.assertIn("inference completed", rendered)
        self.assertIn("frame=11", rendered)
        self.assertIn("latency=0.62s", rendered)


if __name__ == "__main__":
    unittest.main()
