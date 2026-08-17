from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prefmem.agents.hri import HRI_Agent
from prefmem.agents.memory import Memory_Agent
from prefmem.agents.planner import Planner_Agent
from prefmem.runtime import PrefMemRuntime


MODEL = "/models/frozen-gemma"
MODEL_BASE_URL = "http://127.0.0.1:18000/v1"
EMBEDDING_MODEL = "/models/frozen-embeddinggemma"
EMBEDDING_BASE_URL = "http://127.0.0.1:18080/v1"


def model_args(**overrides):
    values = {
        "model": MODEL,
        "model_base_url": MODEL_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_base_url": EMBEDDING_BASE_URL,
        "think": (),
        "print_raw": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _Publisher:
    def reset(self, *, session_id=None) -> None:
        return None

    def publish(self, display) -> None:
        return None


class ModelEndpointConfigurationTests(unittest.TestCase):
    def test_hri_passes_cli_model_identity_to_itself_and_subagents(self) -> None:
        args = model_args()
        model_client = Mock()
        model_client.bind_tools.return_value = Mock()
        tool = _Tool("call_sub_agent")

        with (
            patch(
                "prefmem.agents.hri.VLLMChatOpenAI",
                return_value=model_client,
            ) as model_factory,
            patch(
                "prefmem.agents.hri.StructuredTool.from_function",
                return_value=tool,
            ),
            patch.object(HRI_Agent, "build_agent", return_value=object()),
            patch("prefmem.agents.hri.Planner_Agent") as planner_factory,
            patch("prefmem.agents.hri.Memory_Agent") as memory_factory,
        ):
            agent = HRI_Agent("vllm", args)

        self.assertEqual(agent.config.model, MODEL)
        self.assertEqual(agent.config.model_base_url, MODEL_BASE_URL)
        self.assertEqual(model_factory.call_args.kwargs["model"], MODEL)
        self.assertEqual(
            model_factory.call_args.kwargs["base_url"],
            MODEL_BASE_URL,
        )
        self.assertIs(planner_factory.call_args.kwargs["args"], args)
        self.assertIs(memory_factory.call_args.kwargs["args"], args)

    def test_planner_uses_cli_model_identity(self) -> None:
        args = model_args()
        with (
            patch("prefmem.agents.planner.VLLMChatOpenAI") as model_factory,
            patch.object(Planner_Agent, "build_agent", return_value=object()),
        ):
            agent = Planner_Agent("vllm", args)

        self.assertEqual(agent.config.model, MODEL)
        self.assertEqual(agent.config.model_base_url, MODEL_BASE_URL)
        self.assertEqual(model_factory.call_args.kwargs["model"], MODEL)
        self.assertEqual(
            model_factory.call_args.kwargs["base_url"],
            MODEL_BASE_URL,
        )

    def test_memory_uses_cli_chat_and_embedding_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = model_args(memory_store_path=Path(directory))
            model_client = Mock()
            model_client.bind_tools.return_value = Mock()

            def make_tool(*, func):
                return _Tool(func.__name__)

            with (
                patch(
                    "prefmem.agents.memory.VLLMChatOpenAI",
                    return_value=model_client,
                ) as model_factory,
                patch(
                    "prefmem.agents.memory.VLLMEmbeddingGemma"
                ) as embedding_factory,
                patch(
                    "prefmem.agents.memory.StructuredTool.from_function",
                    side_effect=make_tool,
                ),
                patch.object(Memory_Agent, "build_agent", return_value=object()),
            ):
                agent = Memory_Agent("vllm", args)

        self.assertEqual(model_factory.call_args.kwargs["model"], MODEL)
        self.assertEqual(
            model_factory.call_args.kwargs["base_url"],
            MODEL_BASE_URL,
        )
        embedding_factory.assert_called_once_with(
            model=EMBEDDING_MODEL,
            base_url=EMBEDDING_BASE_URL,
        )
        self.assertEqual(agent.config.embedding_model, EMBEDDING_MODEL)
        self.assertEqual(
            agent.config.embedding_model_base_url,
            EMBEDDING_BASE_URL,
        )

    def test_runtime_passes_model_identity_to_monitor_and_validator(self) -> None:
        planner = SimpleNamespace(preview=Mock(), plan_cycle=Mock())
        monitor = Mock()
        validator = Mock()
        with (
            patch(
                "prefmem.runtime.MonitorService",
                return_value=monitor,
            ) as monitor_factory,
            patch(
                "prefmem.runtime.ValidatorService",
                return_value=validator,
            ) as validator_factory,
        ):
            runtime = PrefMemRuntime(
                planner,
                task_publisher=_Publisher(),
                frame_source=Mock(),
                model_name=MODEL,
                model_base_url=MODEL_BASE_URL,
            )
            runtime.close()

        self.assertEqual(
            monitor_factory.call_args.kwargs["model_name"],
            MODEL,
        )
        self.assertEqual(
            monitor_factory.call_args.kwargs["model_base_url"],
            MODEL_BASE_URL,
        )
        self.assertEqual(
            validator_factory.call_args.kwargs["model_name"],
            MODEL,
        )
        self.assertEqual(
            validator_factory.call_args.kwargs["model_base_url"],
            MODEL_BASE_URL,
        )


if __name__ == "__main__":
    unittest.main()
