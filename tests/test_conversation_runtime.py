from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation.benchmark.conversation_runtime import (
    EvaluationArtifactRecorder,
    EvaluationArtifactSink,
)


class ConversationArtifactRuntimeTests(unittest.TestCase):
    def test_recorder_requires_both_structured_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvaluationArtifactRecorder(directory)
            self.assertFalse(recorder.status()["complete"])
            recorder.model_calls({"kind": "model"})
            recorder.agent_events({"kind": "agent"})
            status = recorder.status()
            self.assertTrue(status["complete"])
            for stream in status["streams"].values():
                self.assertEqual(stream["events_written"], 1)
                self.assertTrue(Path(stream["path"]).is_file())

    def test_sink_records_write_failure_without_interrupting_orchestration(self) -> None:
        sink = EvaluationArtifactSink("unwritten.jsonl")
        with patch(
            "simulation.benchmark.conversation_runtime._append_jsonl",
            side_effect=OSError("disk unavailable"),
        ):
            sink({"event": "still-return"})
        status = sink.status()
        self.assertFalse(status["complete"])
        self.assertEqual(status["error_count"], 1)
        json.dumps(status)


if __name__ == "__main__":
    unittest.main()
