from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from prefmem.cli import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    DEFAULT_TRANSCRIPT,
    DEFAULT_VLLM_BASE_URL,
    parse_args,
)


class CliParserTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = parse_args([])

        self.assertEqual(args.dataset, DEFAULT_DATASET)
        self.assertFalse(args.benchmark)
        self.assertIsNone(args.preference_store)
        self.assertEqual(args.username, "default")
        self.assertEqual(args.transcript, DEFAULT_TRANSCRIPT)
        self.assertFalse(args.display_all)
        self.assertFalse(args.resize_images)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.model_provider, "ollama")
        self.assertIsNone(args.model_base_url)

    def test_path_and_boolean_options(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "packets/episode-1",
                "--benchmark",
                "--preference-store",
                "state/preferences.json",
                "--username",
                "participant-7",
                "--transcript",
                "runs/session.txt",
                "--display-all",
                "--resize-images",
            ]
        )

        self.assertEqual(args.dataset, Path("packets/episode-1"))
        self.assertTrue(args.benchmark)
        self.assertEqual(args.preference_store, Path("state/preferences.json"))
        self.assertEqual(args.username, "participant-7")
        self.assertEqual(args.transcript, Path("runs/session.txt"))
        self.assertTrue(args.display_all)
        self.assertTrue(args.resize_images)

    def test_display_all_spellings_are_equivalent(self) -> None:
        for spelling in ("--display_all", "--display-all"):
            with self.subTest(spelling=spelling):
                self.assertTrue(parse_args([spelling]).display_all)

    def test_memory_store_maps_to_preference_store(self) -> None:
        args = parse_args(["--memory-store", "state/preferences.json"])

        self.assertEqual(args.preference_store, Path("state/preferences.json"))

    def test_memory_store_path_equals_form_selects_requested_directory(self) -> None:
        args = parse_args(["--memory-store-path=./memory_store_test"])

        self.assertEqual(args.preference_store, Path("memory_store_test"))
        self.assertEqual(args.memory_store_path, Path("memory_store_test"))

    def test_memory_store_alias_conflicts_with_canonical_option(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(
                    [
                        "--preference-store",
                        "preferred.json",
                        "--memory-store",
                        "legacy.json",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)

    def test_vllm_uses_default_base_url(self) -> None:
        args = parse_args(["--model-provider", "vllm"])

        self.assertEqual(args.model_base_url, DEFAULT_VLLM_BASE_URL)

    def test_vllm_accepts_valid_custom_base_url(self) -> None:
        args = parse_args(
            [
                "--model-provider",
                "vllm",
                "--model-base-url",
                "https://models.example.test/api/v1",
            ]
        )

        self.assertEqual(
            args.model_base_url,
            "https://models.example.test/api/v1",
        )

    def test_model_base_url_requires_vllm(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(
                    ["--model-base-url", "http://localhost:8000/v1"]
                )

        self.assertEqual(raised.exception.code, 2)

    def test_vllm_rejects_invalid_base_urls(self) -> None:
        invalid_urls = (
            "localhost:8000/v1",
            "ftp://localhost:8000/v1",
            "http://user:secret@localhost:8000/v1",
            "http://localhost:8000/v1/",
            "http://localhost:8000/v1?mode=test",
            "http://localhost:8000/v1#fragment",
        )

        for value in invalid_urls:
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args(
                            [
                                "--model-provider",
                                "vllm",
                                "--model-base-url",
                                value,
                            ]
                        )

                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
