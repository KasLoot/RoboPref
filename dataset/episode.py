from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NUMERIC_FRAME_STEM = re.compile(r"^\d+(?:[_-]\d+)*$")


class DatasetEpisodeError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetEpisode:
    directory: Path
    initial_frame: Path
    final_frame: Path

    @classmethod
    def from_path(cls, path: str | Path, workspace_root: str | Path) -> DatasetEpisode:
        workspace_root = Path(workspace_root).expanduser().resolve()
        directory = Path(path).expanduser()
        if not directory.is_absolute():
            directory = workspace_root / directory
        directory = directory.resolve()

        if not directory.is_dir():
            raise DatasetEpisodeError(f"Dataset directory does not exist: {directory}")

        frames = sorted(
            (
                frame
                for frame in directory.iterdir()
                if frame.is_file()
                and frame.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                and NUMERIC_FRAME_STEM.fullmatch(frame.stem)
            ),
            key=cls._frame_sort_key,
        )
        if len(frames) < 2:
            raise DatasetEpisodeError(
                f"Dataset must contain at least two numerically named image frames: {directory}"
            )

        for frame in (frames[0], frames[-1]):
            try:
                with Image.open(frame) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise DatasetEpisodeError(f"Dataset frame is not a valid image: {frame}") from error

        return cls(directory=directory, initial_frame=frames[0], final_frame=frames[-1])

    def display_path(self, workspace_root: str | Path) -> str:
        workspace_root = Path(workspace_root).expanduser().resolve()
        try:
            return str(self.directory.relative_to(workspace_root))
        except ValueError:
            return str(self.directory)

    @staticmethod
    def _frame_sort_key(frame: Path) -> tuple[tuple[int, ...], str]:
        numeric_parts = tuple(int(part) for part in re.split(r"[_-]", frame.stem))
        return numeric_parts, frame.name