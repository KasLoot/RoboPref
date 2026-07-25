from __future__ import annotations

import unittest

from agents.diagnostics import AgentOutputDisplay


class AgentOutputObserverTests(unittest.TestCase):
    def test_disabled_display_still_emits_sanitized_observer_event(self) -> None:
        events: list[dict[str, object]] = []
        display = AgentOutputDisplay(enabled=False, observer=events.append)

        display.emit(
            "Planner Agent",
            "plan_task raw output",
            {
                "planning_status": "READY",
                "reasoning": "hidden",
                "image": b"private",
            },
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["agent"], "Planner Agent")
        output = event["output"]
        self.assertIsInstance(output, dict)
        assert isinstance(output, dict)
        self.assertEqual(output["reasoning"], "[omitted]")
        self.assertEqual(output["image"]["byte_count"], 7)
        self.assertNotIn("private", repr(event))

    def test_observer_failure_does_not_escape(self) -> None:
        def fail(_event: dict[str, object]) -> None:
            raise OSError("log failure")

        AgentOutputDisplay(observer=fail).emit("HRI Agent", "stage", {"ok": True})


if __name__ == "__main__":
    unittest.main()
