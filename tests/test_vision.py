from __future__ import annotations

import base64
import unittest
from unittest.mock import patch
from urllib.error import URLError

from prefmem.agents.vision import (
    DEFAULT_LIVE_FRAME_URL,
    get_live_frame,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class GetLiveFrameTests(unittest.TestCase):
    @patch("prefmem.agents.vision.urlopen")
    def test_returns_model_compatible_image_block(self, mock_urlopen) -> None:
        jpeg = b"\xff\xd8\xff\xdbtest-jpeg"
        mock_urlopen.return_value = FakeResponse(jpeg)

        frame = get_live_frame()

        expected_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(
            "ascii"
        )
        self.assertEqual(
            frame,
            {
                "type": "image_url",
                "image_url": {"url": expected_url},
            },
        )
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, DEFAULT_LIVE_FRAME_URL)
        self.assertEqual(request.get_header("Accept"), "image/jpeg")
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 5.0)

    @patch("prefmem.agents.vision.urlopen")
    def test_connection_error_has_actionable_message(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = URLError("connection refused")

        with self.assertRaisesRegex(
            RuntimeError,
            "Is the webcam streamer running",
        ):
            get_live_frame()

    @patch("prefmem.agents.vision.urlopen")
    def test_empty_snapshot_is_rejected(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(b"")

        with self.assertRaisesRegex(RuntimeError, "empty frame"):
            get_live_frame()

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            get_live_frame(timeout=0)


if __name__ == "__main__":
    unittest.main()
