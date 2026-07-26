from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main as main_module
from agents.configs import AgentModelConfig, PrefMemConfig
from agents.diagnostics import AgentOutputDisplay, DisplayingJsonModel
from agents.hri import _agent_failure_evidence
from agents.model import (
    JsonModelError,
    OllamaJsonModel,
    VLLMJsonModel,
    VLLM_DEFAULT_BASE_URL,
    build_json_model,
    effective_model_base_url,
)
from simulation.benchmark import __main__ as benchmark_main
from simulation.benchmark.conversation_models import ConversationEvaluationConfig
from simulation.benchmark.model_defaults import (
    DEFAULT_EVALUATION_MODEL,
    DEFAULT_EVALUATION_PROVIDER,
    default_evaluation_prefmem_config,
)


class VLLMJsonModelTests(unittest.TestCase):
    @staticmethod
    def _response(
        content: str | None = '{"mode":"REPORT"}',
        *,
        finish_reason: str = "stop",
    ) -> object:
        return SimpleNamespace(
            id="chatcmpl-test",
            created=123,
            model="served-model",
            system_fingerprint="fp-test",
            service_tier=None,
            _request_id="request-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
        )

    @staticmethod
    def _client(response: object) -> tuple[object, Mock]:
        create = Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )
        return client, create

    def test_multimodal_json_request_and_private_telemetry(self) -> None:
        records: list[dict[str, object]] = []
        client, create = self._client(self._response('prefix {"mode":"REPORT"}'))
        png = b"\x89PNG\r\n\x1a\nprivate-png"
        jpeg = b"\xff\xd8\xffprivate-jpeg"

        with patch("openai.OpenAI", return_value=client) as openai_client:
            model = VLLMJsonModel(
                "configured-model",
                temperature=0.25,
                base_url="http://localhost:8000/v1/",
                api_key="private-key",
                timeout_seconds=9.0,
                seed=42,
                agent_name="HRI Agent",
                telemetry_observer=records.append,
            )
            result = model.generate(
                purpose="resolve_hri_turn",
                system_prompt="private system prompt",
                payload={"user_message": "private request"},
                images=[png, jpeg],
            )

        self.assertEqual(result, {"mode": "REPORT"})
        openai_client.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="private-key",
            timeout=9.0,
            max_retries=0,
        )
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "configured-model")
        self.assertEqual(request["temperature"], 0.25)
        self.assertEqual(request["seed"], 42)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(
            request["messages"][0],
            {"role": "system", "content": "private system prompt"},
        )
        content = request["messages"][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(
            json.loads(content[0]["text"]),
            {"user_message": "private request"},
        )
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        )
        self.assertEqual(
            content[2]["image_url"]["url"],
            "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["success"])
        self.assertEqual(record["provider"], "vllm")
        self.assertEqual(record["image_count"], 2)
        self.assertEqual(
            record["response_metadata"],
            {
                "id": "chatcmpl-test",
                "created": 123,
                "response_model": "served-model",
                "system_fingerprint": "fp-test",
                "request_id": "request-test",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )
        serialized = repr(record)
        for private_value in (
            "private-key",
            "private system prompt",
            "private request",
            "private-png",
            "private-jpeg",
        ):
            self.assertNotIn(private_value, serialized)

    def test_text_only_request_uses_string_content_and_omits_seed(self) -> None:
        client, create = self._client(self._response())
        with patch("openai.OpenAI", return_value=client):
            model = VLLMJsonModel("model")
            result = model.generate(
                purpose="retrieve_history",
                system_prompt="prompt",
                payload={"query": "history"},
            )

        self.assertEqual(result, {"mode": "REPORT"})
        request = create.call_args.kwargs
        self.assertIsInstance(request["messages"][1]["content"], str)
        self.assertNotIn("seed", request)

    def test_truncation_is_observed_and_rejected(self) -> None:
        records: list[dict[str, object]] = []
        client, _create = self._client(
            self._response('{"partial":true}', finish_reason="length")
        )
        with patch("openai.OpenAI", return_value=client):
            model = VLLMJsonModel(
                "model",
                telemetry_observer=records.append,
            )
            with self.assertRaisesRegex(JsonModelError, "truncated"):
                model.generate(
                    purpose="plan_task",
                    system_prompt="prompt",
                    payload={},
                )

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["error_type"], "JsonModelError")

    def test_malformed_completion_is_observed_and_rejected(self) -> None:
        records: list[dict[str, object]] = []
        response = SimpleNamespace(choices=[], usage=None)
        client, _create = self._client(response)
        with patch("openai.OpenAI", return_value=client):
            model = VLLMJsonModel(
                "model",
                telemetry_observer=records.append,
            )
            with self.assertRaisesRegex(JsonModelError, "Malformed vLLM"):
                model.generate(
                    purpose="validate_task",
                    system_prompt="prompt",
                    payload={},
                )

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])

    def test_sdk_error_text_is_sanitized_everywhere_downstream(self) -> None:
        secret = "SECRET server echoed API key and private prompt"
        records: list[dict[str, object]] = []
        display_records: list[dict[str, object]] = []
        create = Mock(side_effect=RuntimeError(secret))
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

        with patch("openai.OpenAI", return_value=client):
            model = VLLMJsonModel(
                "model",
                telemetry_observer=records.append,
            )
            wrapped = DisplayingJsonModel(
                model,
                AgentOutputDisplay(observer=display_records.append),
                "Validator Agent",
            )
            with self.assertRaisesRegex(JsonModelError, "vLLM request failed") as caught:
                wrapped.generate(
                    purpose="validate_task",
                    system_prompt="private prompt",
                    payload={"request": "private request"},
                )

        rendered = "".join(traceback.format_exception(caught.exception))
        failure = _agent_failure_evidence(
            "VALIDATION",
            caught.exception,
            attempt=1,
            event="validator-call",
        )
        self.assertNotIn(secret, repr(display_records))
        self.assertNotIn(secret, repr(failure))
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, rendered)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertNotIn(secret, repr(records[0]))


class ModelProviderFactoryTests(unittest.TestCase):
    def test_config_normalizes_provider_and_base_url(self) -> None:
        config = AgentModelConfig(
            provider=" VLLM ",
            model="served-model",
            base_url="http://localhost:8000/v1/",
        )

        self.assertEqual(config.provider, "vllm")
        self.assertEqual(config.base_url, "http://localhost:8000/v1")

    def test_config_rejects_unsafe_or_non_root_urls(self) -> None:
        invalid_urls = (
            "http://user:secret@localhost:8000/v1",
            "http://localhost:8000/v1?token=secret",
            "http://localhost:8000/v1#secret",
            "http://localhost:8000",
            "http://localhost:8000/v1/chat/completions",
            "ftp://localhost:8000/v1",
            "localhost:8000/v1",
            "http://:8000/v1",
            "http://localhost:notaport/v1",
            "http://localhost:99999/v1",
            "http://localhost:8000/private path/v1",
        )
        for base_url in invalid_urls:
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    AgentModelConfig(
                        provider="vllm",
                        base_url=base_url,
                    )

    def test_config_rejects_provider_specific_option_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires provider"):
            AgentModelConfig(base_url="http://localhost:8000/v1")
        with self.assertRaisesRegex(ValueError, "Ollama host"):
            AgentModelConfig(
                provider="vllm",
                host="http://localhost:11434",
            )

    def test_effective_base_url_reflects_runtime_default(self) -> None:
        vllm_config = AgentModelConfig(provider="vllm")
        self.assertEqual(
            effective_model_base_url(vllm_config),
            VLLM_DEFAULT_BASE_URL,
        )
        self.assertIsNone(effective_model_base_url(AgentModelConfig()))

    def test_factory_selects_ollama(self) -> None:
        model = build_json_model(AgentModelConfig(model="ollama-model"))

        self.assertIsInstance(model, OllamaJsonModel)
        self.assertEqual(model.provider, "ollama")

    def test_factory_selects_vllm_and_reads_dedicated_key_environment(self) -> None:
        config = AgentModelConfig(
            provider="vllm",
            model="served-model",
            base_url=None,
            timeout_seconds=15.0,
        )
        with (
            patch.dict(os.environ, {"VLLM_API_KEY": "local-token"}),
            patch("openai.OpenAI") as openai_client,
        ):
            model = build_json_model(config, agent_name="Planner Agent")

        self.assertIsInstance(model, VLLMJsonModel)
        self.assertEqual(model.base_url, VLLM_DEFAULT_BASE_URL)
        openai_client.assert_called_once_with(
            base_url=VLLM_DEFAULT_BASE_URL,
            api_key="local-token",
            timeout=15.0,
            max_retries=0,
        )

    def test_factory_rejects_mutated_unknown_provider(self) -> None:
        config = AgentModelConfig(model="model")
        config.provider = "unknown"

        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            build_json_model(config)


class VLLMConversationEvaluationTests(unittest.TestCase):
    def test_evaluation_model_defaults_use_local_vllm(self) -> None:
        config = default_evaluation_prefmem_config()
        for name in ("hri", "memory", "planner", "validator"):
            agent_config = getattr(config, name)
            self.assertEqual(agent_config.provider, DEFAULT_EVALUATION_PROVIDER)
            self.assertEqual(agent_config.model, DEFAULT_EVALUATION_MODEL)
            self.assertEqual(
                effective_model_base_url(agent_config),
                VLLM_DEFAULT_BASE_URL,
            )

    def test_conversation_config_uses_shared_url_validation(self) -> None:
        config = ConversationEvaluationConfig(
            benchmark_root=".",
            output_dir="/tmp/robopref-conversation-config",
            model_provider="vllm",
            model_base_url="http://localhost:8000/v1/",
        )
        self.assertEqual(config.model_base_url, VLLM_DEFAULT_BASE_URL)
        with self.assertRaises(ValueError):
            ConversationEvaluationConfig(
                benchmark_root=".",
                output_dir="/tmp/robopref-conversation-config",
                model_provider="vllm",
                model_base_url="http://user:secret@localhost:8000/v1",
            )
        with self.assertRaisesRegex(ValueError, "requires model_provider"):
            ConversationEvaluationConfig(
                benchmark_root=".",
                output_dir="/tmp/robopref-conversation-config",
                model_provider="ollama",
                model_base_url=VLLM_DEFAULT_BASE_URL,
            )

    def test_conversation_cli_preserves_vllm_settings(self) -> None:
        args = benchmark_main._parser().parse_args(
            [
                "evaluate-conversations",
                "dataset/sim_datasets",
                "--output",
                "/tmp/robopref-conversation-output",
                "--model-provider",
                "vllm",
                "--model-base-url",
                "http://localhost:9100/v1/",
                "--model",
                "/models/benchmark-gemma",
            ]
        )
        config = benchmark_main._conversation_config(args)
        self.assertEqual(config.model_provider, "vllm")
        self.assertEqual(config.model_base_url, "http://localhost:9100/v1")
        self.assertEqual(config.model, "/models/benchmark-gemma")

    def test_conversation_cli_defaults_to_vllm(self) -> None:
        args = benchmark_main._parser().parse_args(
            [
                "evaluate-conversations",
                "dataset/sim_datasets",
                "--output",
                "/tmp/robopref-conversation-default",
            ]
        )
        config = benchmark_main._conversation_config(args)
        self.assertEqual(config.model_provider, DEFAULT_EVALUATION_PROVIDER)
        self.assertIsNone(config.model)
        self.assertTrue(config.show_progress)

    def test_conversation_cli_can_disable_progress(self) -> None:
        args = benchmark_main._parser().parse_args(
            [
                "evaluate-conversations",
                "dataset/sim_datasets",
                "--output",
                "/tmp/robopref-conversation-no-progress",
                "--no-progress",
            ]
        )
        config = benchmark_main._conversation_config(args)
        self.assertFalse(config.show_progress)


class VLLMCLIConfigTests(unittest.TestCase):
    def test_interactive_cli_applies_vllm_settings_to_all_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "terminal.txt"
            prefmem_instance = Mock()
            argv = [
                "main.py",
                "--model-provider",
                "vllm",
                "--model-base-url",
                "http://localhost:9000/v1/",
                "--model",
                "/models/local-gemma",
                "--transcript",
                str(transcript),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    main_module,
                    "PrefMem",
                    return_value=prefmem_instance,
                ) as prefmem,
            ):
                main_module.main()

        config = prefmem.call_args.args[0]
        self.assertIsInstance(config, PrefMemConfig)
        for agent_config in (
            config.hri,
            config.memory,
            config.planner,
            config.validator,
        ):
            self.assertEqual(agent_config.provider, "vllm")
            self.assertEqual(agent_config.base_url, "http://localhost:9000/v1")
            self.assertEqual(agent_config.model, "/models/local-gemma")

    def test_interactive_cli_rejects_mismatched_or_invalid_options(self) -> None:
        parser = main_module.build_parser()
        invalid_argv = (
            ["--model-base-url", VLLM_DEFAULT_BASE_URL],
            [
                "--model-provider",
                "vllm",
                "--ollama-host",
                "http://localhost:11434",
            ],
            [
                "--model-provider",
                "vllm",
                "--model-base-url",
                "http://user:secret@localhost:8000/v1",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with self.assertRaises(SystemExit):
                    main_module._validate_model_options(parser, args)

    def test_benchmark_cli_rejects_mismatched_or_invalid_options(self) -> None:
        parser = benchmark_main._parser()
        base = [
            "evaluate-conversations",
            "dataset/sim_datasets",
            "--output",
            "/tmp/robopref-invalid-options",
        ]
        invalid_suffixes = (
            [
                "--model-provider",
                "ollama",
                "--model-base-url",
                VLLM_DEFAULT_BASE_URL,
            ],
            [
                "--model-provider",
                "vllm",
                "--model-base-url",
                "http://user:secret@localhost:8000/v1",
            ],
        )
        for suffix in invalid_suffixes:
            args = parser.parse_args([*base, *suffix])
            with self.assertRaises(SystemExit):
                benchmark_main._validate_model_options(parser, args)


if __name__ == "__main__":
    unittest.main()
