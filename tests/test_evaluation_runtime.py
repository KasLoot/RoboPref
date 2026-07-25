from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation.benchmark.evaluation_runtime import (
    EvaluationArtifactRecorder,
    EvaluationArtifactSink,
)


class EvaluationArtifactRuntimeTests(unittest.TestCase):
    def test_recorder_reports_complete_only_after_both_streams_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvaluationArtifactRecorder(directory)

            self.assertFalse(recorder.status()["complete"])
            recorder.model_calls({"kind": "model"})
            recorder.agent_events({"kind": "agent"})
            status = recorder.status()

            self.assertTrue(status["complete"])
            for stream in status["streams"].values():
                self.assertEqual(stream["events_written"], 1)
                self.assertEqual(stream["error_count"], 0)
                path = Path(stream["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    1,
                )

    def test_sink_retains_write_failure_without_interrupting_agent(self) -> None:
        sink = EvaluationArtifactSink("unwritten.jsonl")

        with patch(
            "simulation.benchmark.evaluation_runtime._append_jsonl",
            side_effect=OSError("disk unavailable"),
        ):
            sink({"event": "still-return"})

        status = sink.status()
        self.assertFalse(status["complete"])
        self.assertEqual(status["events_written"], 0)
        self.assertEqual(status["error_count"], 1)
        self.assertEqual(status["errors"][0]["type"], "OSError")
        # The status is JSON-safe for inclusion in durable trial records.
        json.dumps(status)


if __name__ == "__main__":
    unittest.main()
