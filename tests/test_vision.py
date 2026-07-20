from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from agents.planner import Planner_Agent, Planner_Agent_Config
from agents.vision import VisionImageError, prepare_vision_image
from agents.vlidator import Validator_Agent, Validator_Agent_Config


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class VisionImageTests(unittest.TestCase):
    def test_resize_preserves_full_frame_aspect_ratio_and_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "frame.png"
            source = Image.new("RGB", (800, 400), "red")
            ImageDraw.Draw(source).rectangle((400, 0, 799, 399), fill="blue")
            source.save(source_path)
            original_bytes = source_path.read_bytes()

            encoded = prepare_vision_image(source_path)

            self.assertEqual(source_path.read_bytes(), original_bytes)
            self.assertTrue(encoded.startswith(b"\xff\xd8"))
            with Image.open(BytesIO(encoded)) as prepared:
                self.assertEqual(prepared.size, (640, 320))
                left_pixel = prepared.getpixel((50, 160))
                right_pixel = prepared.getpixel((590, 160))
                self.assertGreater(left_pixel[0], left_pixel[2])
                self.assertGreater(right_pixel[2], right_pixel[0])

    def test_resize_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "frame.png"
            Image.new("RGB", (800, 400), "green").save(source_path)

            encoded = prepare_vision_image(source_path, resize=False)

            with Image.open(BytesIO(encoded)) as prepared:
                self.assertEqual(prepared.size, (800, 400))

    def test_invalid_image_error_contains_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "broken.png"
            source_path.write_text("not an image", encoding="utf-8")

            with self.assertRaises(VisionImageError) as context:
                prepare_vision_image(source_path)

            self.assertIn(str(source_path.resolve()), str(context.exception))

    def test_large_repository_frame_is_bounded_below_one_megabyte(self):
        source_path = WORKSPACE_ROOT / "dataset" / "v5" / "1.jpg"

        encoded = prepare_vision_image(source_path)

        self.assertLess(len(encoded), 1_000_000)
        with Image.open(BytesIO(encoded)) as prepared:
            self.assertLessEqual(prepared.width, 640)
            self.assertLessEqual(prepared.height, 480)


class VisualAgentPayloadTests(unittest.TestCase):
    def test_planner_sends_prepared_image_bytes(self):
        config = Planner_Agent_Config()
        agent = Planner_Agent(config)
        prepared = b"prepared planner image"

        with (
            patch("agents.planner.prepare_vision_image", return_value=prepared) as prepare,
            patch("agents.planner.chat", return_value=[]) as chat,
        ):
            agent.plan("Stack the blocks", "initial.jpg")

        prepare.assert_called_once_with(
            "initial.jpg",
            resize=True,
            width=640,
            height=480,
            jpeg_quality=85,
        )
        self.assertEqual(chat.call_args.kwargs["messages"][1]["images"], [prepared])

    def test_validator_sends_prepared_image_bytes(self):
        config = Validator_Agent_Config()
        config.resize_images = False
        agent = Validator_Agent(config)
        prepared = b"prepared validator image"

        with (
            patch("agents.vlidator.prepare_vision_image", return_value=prepared) as prepare,
            patch("agents.vlidator.chat", return_value=[]) as chat,
        ):
            agent.validate("Stack the blocks", "final.jpg")

        prepare.assert_called_once_with(
            "final.jpg",
            resize=False,
            width=640,
            height=480,
            jpeg_quality=85,
        )
        self.assertEqual(chat.call_args.kwargs["messages"][1]["images"], [prepared])


if __name__ == "__main__":
    unittest.main()
