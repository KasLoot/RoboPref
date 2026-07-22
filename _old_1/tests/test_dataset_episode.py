import tempfile
import unittest
from pathlib import Path

from dataset.episode import DatasetEpisode, DatasetEpisodeError


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class DatasetEpisodeTests(unittest.TestCase):
    def test_v3_uses_first_and_last_numeric_png_frames(self):
        episode = DatasetEpisode.from_path("dataset/v3", WORKSPACE_ROOT)

        self.assertEqual(episode.initial_frame.name, "1.png")
        self.assertEqual(episode.final_frame.name, "12.png")

    def test_v5_supports_jpg_and_compound_numeric_stems(self):
        episode = DatasetEpisode.from_path("dataset/v5", WORKSPACE_ROOT)

        self.assertEqual(episode.initial_frame.name, "1.jpg")
        self.assertEqual(episode.final_frame.name, "3_2.jpg")

    def test_relative_paths_resolve_from_workspace_root(self):
        episode = DatasetEpisode.from_path("dataset/v3", WORKSPACE_ROOT)

        self.assertEqual(episode.directory, WORKSPACE_ROOT / "dataset" / "v3")
        self.assertEqual(episode.display_path(WORKSPACE_ROOT), "dataset/v3")

    def test_directory_requires_two_numeric_image_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "first.jpg").touch()
            (directory / "1.jpg").touch()

            with self.assertRaisesRegex(DatasetEpisodeError, "at least two"):
                DatasetEpisode.from_path(directory, WORKSPACE_ROOT)

    def test_selected_frames_must_be_valid_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "1.jpg").write_text("not an image")
            (directory / "2.jpg").write_text("not an image")

            with self.assertRaisesRegex(DatasetEpisodeError, "not a valid image"):
                DatasetEpisode.from_path(directory, WORKSPACE_ROOT)


if __name__ == "__main__":
    unittest.main()