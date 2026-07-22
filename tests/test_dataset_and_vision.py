from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from agents.vision import VisionImageError, prepare_vision_image
from dataset.episode import DatasetEpisode, DatasetEpisodeError


ROOT = Path(__file__).resolve().parents[1]


class DatasetEpisodeTests(unittest.TestCase):
    def test_real_v3_uses_numeric_first_and_last_frames(self) -> None:
        episode = DatasetEpisode.from_path("dataset/v3", ROOT)
        self.assertEqual(episode.initial_frame.name, "1.png")
        self.assertEqual(episode.final_frame.name, "12.png")
        self.assertEqual(episode.display_path(ROOT), "dataset/v3")

    def test_numeric_sort_ignores_non_frame_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, colour in (
                ("10.png", "blue"),
                ("2.png", "green"),
                ("1.png", "red"),
                ("poster.png", "white"),
            ):
                Image.new("RGB", (8, 6), colour).save(path / name)

            episode = DatasetEpisode.from_path(path, ROOT)

            self.assertEqual(episode.initial_frame.name, "1.png")
            self.assertEqual(episode.final_frame.name, "10.png")

    def test_episode_requires_two_numeric_valid_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new("RGB", (8, 6), "red").save(path / "1.png")
            Image.new("RGB", (8, 6), "blue").save(path / "final.png")
            with self.assertRaisesRegex(DatasetEpisodeError, "at least two"):
                DatasetEpisode.from_path(path, ROOT)


class VisionPreparationTests(unittest.TestCase):
    def test_disabled_resize_preserves_lossless_source_bytes_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transparent.png"
            Image.new("RGBA", (23, 11), (255, 0, 0, 0)).save(source)
            original = source.read_bytes()

            encoded = prepare_vision_image(source, resize=False)

            self.assertEqual(encoded, original)
            with Image.open(BytesIO(encoded)) as prepared:
                prepared.load()
                self.assertEqual(prepared.format, "PNG")
                self.assertEqual(prepared.mode, "RGBA")
                self.assertEqual(prepared.size, (23, 11))

    def test_resize_uses_aspect_preserving_thumbnail_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide.png"
            Image.new("RGB", (100, 50), "purple").save(source)

            encoded = prepare_vision_image(source, resize=True, width=20, height=20)

            with Image.open(BytesIO(encoded)) as prepared:
                self.assertEqual(prepared.size, (20, 10))

    def test_invalid_resize_configuration_fails_before_model_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frame.png"
            Image.new("RGB", (10, 10), "black").save(source)
            with self.assertRaisesRegex(VisionImageError, "positive"):
                prepare_vision_image(source, resize=True, width=0, height=10)


if __name__ == "__main__":
    unittest.main()
