from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from prefmem.agents.hri import HRI_Agent
from prefmem.agents.metrics import TurnMetrics
from prefmem.agents.planner import Planner_Agent


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class RecordingModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list] = []

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        self.calls.append(messages)
        return self.response


class RecordingMetrics:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **record) -> None:
        self.records.append(record)


class FakePlanner:
    def run(self, messages: list) -> dict:
        return {
            "messages": [
                messages[0],
                AIMessage(content="planner result"),
            ]
        }


class SinglePassGraph:
    def __init__(self, final_state: dict) -> None:
        self.final_state = final_state
        self.stream_calls = 0
        self.stream_kwargs: dict | None = None

    def stream(self, **kwargs):
        self.stream_calls += 1
        self.stream_kwargs = kwargs
        yield {
            "type": "values",
            "ns": (),
            "data": self.final_state,
        }


class AgentMetricsTests(unittest.TestCase):
    def test_hri_records_usage_at_model_boundary(self) -> None:
        answer = AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )
        model = RecordingModel(answer)
        metrics = RecordingMetrics()
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="system")
        hri.hri_llm = model
        hri.metrics = metrics
        hri.thinking_enabled = False

        result = hri.llm_call(
            {
                "messages": [HumanMessage(content="question")],
                "llm_calls": 0,
            },
            SimpleNamespace(context={"current_frame": {"type": "image_url"}}),
        )

        self.assertIs(result["messages"][0], answer)
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(metrics.records[0]["agent"], "HRI Agent")
        self.assertIs(metrics.records[0]["response"], answer)
        self.assertIsInstance(
            metrics.records[0]["prompt_messages"][0],
            SystemMessage,
        )

    def test_planner_records_usage_at_model_boundary(self) -> None:
        answer = AIMessage(
            content="plan",
            usage_metadata={
                "input_tokens": 8,
                "output_tokens": 3,
                "total_tokens": 11,
            },
        )
        model = RecordingModel(answer)
        metrics = RecordingMetrics()
        planner = object.__new__(Planner_Agent)
        planner.config = SimpleNamespace(system_prompt="planner system")
        planner.llm = model
        planner.metrics = metrics
        planner.thinking_enabled = False

        result = planner.llm_call(
            {
                "messages": [HumanMessage(content="internal request")],
                "llm_calls": 0,
            },
            SimpleNamespace(context={"current_frame": {"type": "image_url"}}),
        )

        self.assertIs(result["messages"][0], answer)
        self.assertEqual(len(metrics.records), 1)
        self.assertEqual(metrics.records[0]["agent"], "Planner Agent")

    def test_summary_attributes_tokens_and_uses_server_throughput(self) -> None:
        metrics = TurnMetrics(WordCounter())

        metrics.record(
            agent="HRI Agent",
            prompt_messages=[
                SystemMessage(content="hri system"),
                HumanMessage(content="user query"),
            ],
            response=AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "call_sub_agent",
                        "args": {"message": "internal request"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                },
                response_metadata={
                    "vllm_metrics": {"tokens_per_second": 3.0}
                },
            ),
            elapsed_seconds=2.0,
        )
        metrics.record(
            agent="Planner Agent",
            prompt_messages=[
                SystemMessage(content="planner system"),
                HumanMessage(content="internal request"),
            ],
            response=AIMessage(
                content="plan",
                usage_metadata={
                    "input_tokens": 8,
                    "output_tokens": 5,
                    "total_tokens": 13,
                },
                response_metadata={
                    "vllm_metrics": {"tokens_per_second": 5.0}
                },
            ),
            elapsed_seconds=2.0,
        )
        metrics.record(
            agent="HRI Agent",
            prompt_messages=[
                SystemMessage(content="hri system"),
                HumanMessage(content="user query"),
            ],
            response=AIMessage(
                content="final answer",
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                },
                response_metadata={
                    "vllm_metrics": {"tokens_per_second": 4.0}
                },
            ),
            elapsed_seconds=2.0,
        )

        summary = metrics.summary(turn_seconds=7.0)
        total = summary["total"]

        self.assertEqual(total["system_prompt_tokens"], 6)
        self.assertEqual(total["query_prompt_tokens"], 4)
        self.assertEqual(total["internal_query_tokens"], 2)
        self.assertEqual(total["internal_query_generated_tokens"], 3)
        self.assertEqual(total["internal_generated_tokens"], 5)
        self.assertEqual(total["hri_generated_tokens"], 4)
        self.assertEqual(total["input_tokens"], 38)
        self.assertEqual(total["output_tokens"], 12)
        self.assertEqual(total["total_tokens"], 50)
        self.assertEqual(total["throughput_source"], "vllm")
        self.assertAlmostEqual(
            total["throughput_tokens_per_second"],
            4.0,
        )

        self.assertEqual(
            summary["agents"]["HRI Agent"]["generated_tokens"],
            7,
        )
        self.assertEqual(
            summary["agents"]["Planner Agent"]["generated_tokens"],
            5,
        )

    def test_planner_streams_graph_once(self) -> None:
        final_state = {
            "messages": [AIMessage(content="plan")],
            "llm_calls": 1,
        }
        graph = SinglePassGraph(final_state)
        planner = object.__new__(Planner_Agent)
        planner.agent = graph
        planner.print_raw = False

        with contextlib.redirect_stdout(io.StringIO()):
            result = planner.invoke_agent(
                [HumanMessage(content="request")],
                current_frame={},
            )

        self.assertIs(result, final_state)
        self.assertEqual(graph.stream_calls, 1)
        self.assertIn("values", graph.stream_kwargs["stream_mode"])

    def test_sub_agent_returns_only_planner_text(self) -> None:
        hri = object.__new__(HRI_Agent)
        hri.planner_agent = FakePlanner()

        result = hri.call_sub_agent(
            sub_agent_name="PLANNER_AGENT",
            message="make a plan",
        )

        self.assertEqual(result, "planner result")


if __name__ == "__main__":
    unittest.main()
