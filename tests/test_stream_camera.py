from __future__ import annotations

import unittest

from prefmem.stream_camera import (
    backend_id,
    build_parser,
    configure_capture,
    fourcc_name,
)


class FakeCv2:
    CAP_ANY = 0
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BUFFERSIZE = 38

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        return sum(
            ord(character) << (8 * index)
            for index, character in enumerate(characters)
        )


class FakeCapture:
    def __init__(self) -> None:
        self.settings: list[tuple[int, int]] = []

    def set(self, property_id: int, value: int) -> bool:
        self.settings.append((property_id, value))
        return True


class StreamCameraConfigurationTests(unittest.TestCase):
    def test_system_flag_accepts_linux_and_windows(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.parse_args(["--system", "linux"]).system, "linux")
        self.assertEqual(parser.parse_args(["--system", "windows"]).system, "windows")

    def test_linux_uses_v4l2_and_requests_mjpeg_first(self) -> None:
        capture = FakeCapture()
        mjpg = FakeCv2.VideoWriter_fourcc(*"MJPG")

        configure_capture(
            FakeCv2,
            capture,
            system="linux",
            width=1280,
            height=720,
            fps=30,
        )

        self.assertEqual(backend_id(FakeCv2, "linux", "auto"), FakeCv2.CAP_V4L2)
        self.assertEqual(
            capture.settings,
            [
                (FakeCv2.CAP_PROP_FOURCC, mjpg),
                (FakeCv2.CAP_PROP_FRAME_WIDTH, 1280),
                (FakeCv2.CAP_PROP_FRAME_HEIGHT, 720),
                (FakeCv2.CAP_PROP_FPS, 30),
                (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
            ],
        )
        self.assertEqual(fourcc_name(mjpg), "MJPG")

    def test_windows_preserves_existing_capture_configuration(self) -> None:
        capture = FakeCapture()

        configure_capture(
            FakeCv2,
            capture,
            system="windows",
            width=1280,
            height=720,
            fps=30,
        )

        self.assertEqual(
            backend_id(FakeCv2, "windows", "dshow"),
            FakeCv2.CAP_DSHOW,
        )
        self.assertEqual(
            capture.settings,
            [
                (FakeCv2.CAP_PROP_FRAME_WIDTH, 1280),
                (FakeCv2.CAP_PROP_FRAME_HEIGHT, 720),
                (FakeCv2.CAP_PROP_FPS, 30),
                (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
