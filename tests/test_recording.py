from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage

from prefmem.recording import MarkdownExperimentRecorder


class MarkdownExperimentRecorderTests(unittest.TestCase):
    def test_dot_segment_name_cannot_escape_recording_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recordings"
            recorder = MarkdownExperimentRecorder(root, session_name="..")

            self.assertEqual(recorder.session_dir.parent, root.resolve())
            self.assertEqual(
                stat.S_IMODE(recorder.session_dir.stat().st_mode),
                0o700,
            )
            self.assertEqual(
                stat.S_IMODE(recorder.markdown_path.stat().st_mode),
                0o600,
            )

    def test_records_context_and_saves_frames_without_base64_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = MarkdownExperimentRecorder(
                Path(temporary),
                session_name="test-session",
                configuration={"camera": "localhost"},
            )
            frame = b"\xff\xd8\xff\xdbtest-frame"
            frame_path = recorder.record_frame(
                frame,
                source="hri-input",
                media_type="image/jpeg",
            )
            self.assertEqual(
                stat.S_IMODE(frame_path.stat().st_mode),
                0o600,
            )
            image_block = {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/very-large-payload"
                },
            }
            recorder.record_user("Where is the banana?", frame_path=frame_path)
            recorder.record_model_exchange(
                agent="HRI",
                messages=[
                    SystemMessage(content="system contract"),
                    HumanMessage(
                        content=[
                            image_block,
                            {"type": "text", "text": "Where is the banana?"},
                        ]
                    ),
                ],
                response={"decision": "RESPOND"},
                frame_paths=[frame_path],
            )
            recorder.record_assistant("The banana is in the middle.")

            markdown = recorder.markdown_path.read_text(encoding="utf-8")
            self.assertEqual(frame_path.read_bytes(), frame)
            self.assertIn("Where is the banana?", markdown)
            self.assertIn("The banana is in the middle.", markdown)
            self.assertIn("frames/frame-00001-hri-input.jpg", markdown)
            self.assertIn("<saved frame; see frame_paths>", markdown)
            self.assertNotIn("very-large-payload", markdown)


if __name__ == "__main__":
    unittest.main()
