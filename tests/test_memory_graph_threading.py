from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from langchain.messages import AIMessage

from prefmem.agents.config import Memory_Config
from prefmem.agents.memory import Memory_Agent, _parse_memory_json_object


class _Graph:
    def __init__(self) -> None:
        self.configs: list[dict] = []

    def stream(self, *, input, config, context, stream_mode, version):
        del input, context, stream_mode, version
        self.configs.append(config)
        yield {
            "type": "values",
            "data": {"messages": [AIMessage(content="done")], "llm_calls": 1},
        }


class MemoryGraphThreadingTests(unittest.TestCase):
    def test_production_candidate_window_matches_ab_mem_q_selection(self) -> None:
        config = Memory_Config("vllm")
        self.assertEqual((config.top_k, config.top_cap_k), (3, 3))

    def test_memory_json_parser_accepts_one_json_fence_only(self) -> None:
        payload = _parse_memory_json_object(
            '```json\n{"status":"EMPTY","retrieved_memory":[]}\n```'
        )
        self.assertEqual(payload["status"], "EMPTY")

        with self.assertRaisesRegex(ValueError, "Memory retrieval response"):
            _parse_memory_json_object(
                'prefix\n```json\n{"status":"EMPTY"}\n```'
            )

    def test_each_request_uses_its_request_id_as_langgraph_thread(self) -> None:
        memory = object.__new__(Memory_Agent)
        graph = _Graph()
        memory.agent = graph

        with redirect_stdout(io.StringIO()):
            memory._request_id = "request-one"
            first = memory.invoke_agent([], {})
            memory._request_id = "request-two"
            second = memory.invoke_agent([], {})

        self.assertEqual(first["messages"][-1].content, "done")
        self.assertEqual(second["messages"][-1].content, "done")
        self.assertEqual(
            [item["configurable"]["thread_id"] for item in graph.configs],
            ["memory-request-request-one", "memory-request-request-two"],
        )
        self.assertTrue(
            all(item["recursion_limit"] == 20 for item in graph.configs)
        )


if __name__ == "__main__":
    unittest.main()
