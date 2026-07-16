import builtins
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transcript import TerminalTranscript


class TerminalTranscriptTests(unittest.TestCase):
    def test_records_prompts_input_stdout_and_stderr_without_ansi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiments" / "record.txt"
            terminal_stdout = io.StringIO()
            terminal_stderr = io.StringIO()

            with (
                patch.object(sys, "stdout", terminal_stdout),
                patch.object(sys, "stderr", terminal_stderr),
                patch.object(builtins, "input", side_effect=lambda prompt="": (print(prompt, end=""), "dataset/v5")[1]),
            ):
                with TerminalTranscript(path):
                    print("\x1b[32mValidator complete\x1b[0m")
                    selected = input("Dataset: ")
                    print(f"Selected {selected}", file=sys.stderr)

            transcript = path.read_text(encoding="utf-8")
            self.assertIn("Validator complete", transcript)
            self.assertIn("Dataset: dataset/v5\n", transcript)
            self.assertIn("Selected dataset/v5", transcript)
            self.assertNotIn("\x1b[", transcript)
            self.assertIn("\x1b[32mValidator complete\x1b[0m", terminal_stdout.getvalue())

    def test_appends_sessions_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.txt"
            with TerminalTranscript(path):
                print("first run")
            with TerminalTranscript(path):
                print("second run")

            transcript = path.read_text(encoding="utf-8")
            self.assertIn("first run", transcript)
            self.assertIn("second run", transcript)
            self.assertEqual(transcript.count("=== Session "), 2)

    def test_records_an_exception_without_suppressing_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.txt"

            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                with TerminalTranscript(path):
                    raise RuntimeError("model unavailable")

            transcript = path.read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last):", transcript)
            self.assertIn("RuntimeError: model unavailable", transcript)


if __name__ == "__main__":
    unittest.main()