from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from prefmem.cli import (
    DEFAULT_DATASET,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
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
        self.assertIsNone(args.simulation_scene)
        self.assertFalse(args.display_all)
        self.assertFalse(args.resize_images)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.model_provider, DEFAULT_MODEL_PROVIDER)
        self.assertEqual(args.model_base_url, DEFAULT_VLLM_BASE_URL)
        self.assertEqual(args.embedding_model, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(
            args.embedding_model_base_url,
            DEFAULT_EMBEDDING_BASE_URL,
        )
        self.assertTrue(args.auto_timeout_replan)
        self.assertEqual(args.max_consecutive_timeout_replans, 2)
        self.assertEqual(args.max_timeout_replans_per_instruction, 2)
        self.assertFalse(args.experiment_oracle_grounding_fallback)
        self.assertEqual(args.monitor_events, "off")

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

    def test_mujoco_executor_options(self) -> None:
        args = parse_args(
            [
                "--executor",
                "mujoco",
                "--sam-base-url",
                "http://127.0.0.1:9000",
                "--sam-threshold",
                "0.65",
                "--simulation-seed",
                "17",
                "--no-simulation-viewer",
                "--simulation-render-size",
                "320",
            ]
        )
        self.assertEqual(args.executor, "mujoco")
        self.assertEqual(args.sam_threshold, 0.65)
        self.assertEqual(args.simulation_seed, 17)
        self.assertFalse(args.simulation_viewer)
        self.assertEqual(args.simulation_render_size, 320)
        self.assertEqual(args.simulation_viewer_camera, "overview")
        self.assertEqual(args.monitor_events, "summary")

    def test_custom_simulation_scene_can_be_selected(self) -> None:
        args = parse_args(["--simulation-scene", "scenes/custom.xml"])

        self.assertEqual(args.simulation_scene, Path("scenes/custom.xml"))

    def test_fixed_task_camera_can_be_selected_explicitly(self) -> None:
        args = parse_args(
            ["--executor", "mujoco", "--simulation-viewer-camera", "task"]
        )

        self.assertEqual(args.simulation_viewer_camera, "task")

    def test_each_dual_camera_view_can_be_inspected_explicitly(self) -> None:
        for camera in ("sam", "prefmem"):
            with self.subTest(camera=camera):
                args = parse_args(
                    [
                        "--executor",
                        "mujoco",
                        "--simulation-viewer-camera",
                        camera,
                    ]
                )
                self.assertEqual(args.simulation_viewer_camera, camera)

    def test_explicit_monitor_event_verbosity_is_preserved(self) -> None:
        args = parse_args(
            ["--executor", "mujoco", "--monitor-events", "verbose"]
        )

        self.assertEqual(args.monitor_events, "verbose")

    def test_sam_threshold_rejects_values_outside_probability_range(self) -> None:
        for value in ("-0.01", "1.01"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--sam-threshold", value])

    def test_sam_endpoint_must_remain_on_loopback(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--sam-base-url", "https://sam.example.test"])

    def test_oracle_grounding_fallback_is_explicit_and_mujoco_only(self) -> None:
        args = parse_args(
            ["--executor", "mujoco", "--experiment-oracle-grounding-fallback"]
        )
        self.assertTrue(args.experiment_oracle_grounding_fallback)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--experiment-oracle-grounding-fallback"])

    def test_simulation_render_size_rejects_oversized_framebuffer(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--simulation-render-size", "1025"])

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
                    [
                        "--model-provider",
                        "ollama",
                        "--model-base-url",
                        "http://localhost:8000/v1",
                    ]
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

    def test_embedding_endpoint_and_timeout_ablation_options(self) -> None:
        args = parse_args(
            [
                "--embedding-model",
                "/models/embedding-test",
                "--embedding-model-base-url",
                "http://127.0.0.1:18080/v1",
                "--no-auto-timeout-replan",
                "--max-consecutive-timeout-replans",
                "4",
                "--max-timeout-replans-per-instruction",
                "3",
            ]
        )

        self.assertEqual(args.embedding_model, "/models/embedding-test")
        self.assertEqual(
            args.embedding_model_base_url,
            "http://127.0.0.1:18080/v1",
        )
        self.assertFalse(args.auto_timeout_replan)
        self.assertEqual(args.max_consecutive_timeout_replans, 4)
        self.assertEqual(args.max_timeout_replans_per_instruction, 3)

    def test_embedding_and_timeout_options_are_validated(self) -> None:
        invalid_arguments = (
            ["--embedding-model", "   "],
            ["--embedding-model-base-url", "localhost:8080/v1"],
            ["--max-consecutive-timeout-replans", "0"],
            ["--max-timeout-replans-per-instruction", "0"],
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
