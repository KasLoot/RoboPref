from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from prefmem.agents.contracts import ExecutionCommand
from prefmem.execution import (
    CallbackVLAAdapter,
    IdempotentVLAAdapter,
    JsonlVLAAdapter,
)


def command(instruction: str = "Move the banana.") -> ExecutionCommand:
    return ExecutionCommand(
        dispatch_id="dispatch-1",
        subtask_id="subtask-1",
        task_instruction=instruction,
        observation_id="obs-1",
        current_frame={"type": "image_url"},
    )


class IdempotentVLAAdapterTests(unittest.TestCase):
    def test_duplicate_dispatch_is_suppressed(self) -> None:
        published = []
        adapter = IdempotentVLAAdapter(CallbackVLAAdapter(published.append))

        first = adapter.publish(command())
        duplicate = adapter.publish(command())

        self.assertEqual(len(published), 1)
        self.assertTrue(first.published)
        self.assertTrue(duplicate.duplicate_suppressed)

    def test_conflicting_dispatch_reuse_is_rejected(self) -> None:
        adapter = IdempotentVLAAdapter(CallbackVLAAdapter(lambda _: None))
        adapter.publish(command())

        with self.assertRaisesRegex(ValueError, "reused"):
            adapter.publish(command("Move the mug."))

    def test_jsonl_adapter_publishes_complete_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.jsonl"
            adapter = JsonlVLAAdapter(path)

            adapter.publish(command())

            text = path.read_text(encoding="utf-8")
            self.assertIn('"dispatch_id": "dispatch-1"', text)
            self.assertIn('"current_frame"', text)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_jsonl_idempotence_survives_adapter_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.jsonl"
            JsonlVLAAdapter(path).publish(command())

            restarted = JsonlVLAAdapter(path)
            duplicate = restarted.publish(command())

            self.assertTrue(duplicate.duplicate_suppressed)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "reused"):
                restarted.publish(command("Move the mug."))

    def test_jsonl_cancellation_is_durable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.jsonl"
            adapter = JsonlVLAAdapter(path)
            adapter.publish(command())
            first = adapter.cancel("dispatch-1", "monitor safety abort")

            restarted = JsonlVLAAdapter(path)
            duplicate = restarted.cancel(
                "dispatch-1",
                "monitor safety abort",
            )

            self.assertTrue(first.supported)
            self.assertTrue(first.accepted)
            self.assertTrue(duplicate.duplicate_suppressed)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"message_type": "CANCEL"', lines[1])

    def test_callback_adapter_reports_optional_cancellation_support(self) -> None:
        published = []
        unsupported = CallbackVLAAdapter(published.append)

        receipt = unsupported.cancel("dispatch-1", "terminal stop")

        self.assertFalse(receipt.supported)
        self.assertFalse(receipt.accepted)

        cancellations = []
        supported = IdempotentVLAAdapter(
            CallbackVLAAdapter(
                published.append,
                cancel_callback=lambda dispatch_id, reason: cancellations.append(
                    (dispatch_id, reason)
                ),
            )
        )
        supported.publish(command())
        first = supported.cancel("dispatch-1", "terminal stop")
        duplicate = supported.cancel("dispatch-1", "terminal stop")

        self.assertEqual(cancellations, [("dispatch-1", "terminal stop")])
        self.assertTrue(first.accepted)
        self.assertTrue(duplicate.duplicate_suppressed)


if __name__ == "__main__":
    unittest.main()
