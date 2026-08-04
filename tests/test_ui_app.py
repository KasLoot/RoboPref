from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from ui.ui import (
    DEFAULT_CAMERA_BASE_URL,
    DEFAULT_GUI_HOST,
    DEFAULT_GUI_PORT,
    OperatorDashboard,
    OperatorConfig,
    build_runtime_factory,
    parse_args,
)


class OperatorAppTests(unittest.TestCase):
    def test_defaults_use_loopback_and_avoid_embedding_port(self):
        config = parse_args([])

        self.assertEqual(DEFAULT_GUI_HOST, "127.0.0.1")
        self.assertEqual(config.camera_base_url, DEFAULT_CAMERA_BASE_URL)
        self.assertEqual(config.port, DEFAULT_GUI_PORT)
        self.assertNotEqual(config.port, 8080)
        self.assertEqual(config.think, ("HRI",))

    def test_camera_origin_rejects_credentials_and_paths(self):
        for value in (
            "http://user:secret@127.0.0.1:1234",
            "http://127.0.0.1:1234/snapshot.jpg",
            "http://127.0.0.1:99999",
            'http://127.0.0.1:1234" autofocus="true',
            "http://127.0.0.1:1234\nmalicious",
            "http://bad_host:1234",
            "not-a-url",
        ):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    parse_args(["--camera-base-url", value])

    def test_camera_origin_is_canonicalized(self):
        config = parse_args(
            ["--camera-base-url", "HTTP://LOCALHOST:1234/"]
        )
        ipv6 = parse_args(
            ["--camera-base-url", "http://[0:0:0:0:0:0:0:1]:1234"]
        )

        self.assertEqual(config.camera_base_url, "http://localhost:1234")
        self.assertEqual(ipv6.camera_base_url, "http://[::1]:1234")

    def test_runtime_factory_is_lazy_and_forces_documented_vllm(self):
        config = OperatorConfig(
            camera_base_url="http://127.0.0.1:4321",
            memory_store_path=Path("custom-memory"),
            think=(),
        )
        runtime = object()
        hri = object()

        with (
            patch("prefmem.cli.parse_args", return_value="parsed") as parse,
            patch(
                "prefmem.runtime.build_runtime",
                return_value=(runtime, hri),
            ) as build,
        ):
            factory = build_runtime_factory(config)
            parse.assert_not_called()
            build.assert_not_called()

            self.assertEqual(factory(), (runtime, hri))

        argv = parse.call_args.args[0]
        self.assertIn("vllm", argv)
        self.assertIn("http://127.0.0.1:4321", argv)
        self.assertIn("custom-memory", argv)
        build.assert_called_once_with("parsed")

    def test_agent_status_labels_cover_streaming_and_startup_states(self):
        self.assertEqual(
            OperatorDashboard._agent_status_meta("working"),
            ("working", "Working"),
        )
        self.assertEqual(
            OperatorDashboard._agent_status_meta("watching"),
            ("working", "Watching"),
        )
        self.assertEqual(
            OperatorDashboard._agent_status_meta("starting"),
            ("working", "Starting"),
        )

    def test_trace_sections_keep_reasoning_and_tool_details_collapsible(self):
        sections = OperatorDashboard._trace_sections(
            [
                {
                    "kind": "thinking_delta",
                    "node": "HRI Agent",
                    "content": "Inspect preference. ",
                },
                {
                    "kind": "tool_call",
                    "node": "HRI Agent",
                    "payload": {
                        "name": "retrieve_preferences",
                        "arguments": {"query": "placement"},
                    },
                },
                {
                    "kind": "tool_result",
                    "node": "Tool Node",
                    "content": "Use the left side.",
                },
                {
                    "kind": "assistant_delta",
                    "node": "HRI Agent",
                    "content": "I will use the left side.",
                },
            ]
        )

        self.assertEqual(sections[0], (
            "Thinking",
            "HRI Agent",
            "Inspect preference. ",
        ))
        self.assertEqual(sections[1][0], "Tool call")
        self.assertIn("retrieve_preferences", sections[1][2])
        self.assertEqual(
            sections[2],
            ("Tool result", "Tool Node", "Use the left side."),
        )
        self.assertNotIn("I will use the left side.", str(sections))


if __name__ == "__main__":
    unittest.main()
