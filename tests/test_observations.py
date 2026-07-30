from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prefmem.observations import DatasetObservationSource


class DatasetObservationSourceTests(unittest.TestCase):
    def test_dataset_frames_use_natural_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            for name in (
                "frame-10.jpg",
                "frame-2.jpg",
                "frame-1.jpg",
                "other.jpg",
            ):
                (dataset / name).touch()

            source = DatasetObservationSource(dataset)

            self.assertEqual(
                [path.name for path in source.files],
                ["frame-1.jpg", "frame-2.jpg", "frame-10.jpg", "other.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
