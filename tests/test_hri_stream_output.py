from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages.tool import tool_call_chunk

from prefmem.agents.hri import HRI_Agent


class FakeStreamingAgent:
    def __init__(self, parts: list[dict], final_state: dict) -> None:
        self.parts = parts
        self.final_state = final_state

    def stream(self, **kwargs):
        yield from self.parts

    def get_state(self, config):
        return SimpleNamespace(values=self.final_state)


def message_part(chunk: AIMessageChunk) -> dict:
    return {
        "type": "messages",
        "ns": (),
        "data": (chunk, {"langgraph_node": "HRI Agent"}),
    }


def update_part(node_name: str, messages: list) -> dict:
    return {
        "type": "updates",
        "ns": (),
        "data": {node_name: {"messages": messages}},
    }


class HRIStreamOutputTests(unittest.TestCase):
    def make_hri(self, parts: list[dict], final_message: AIMessage, *, print_raw=False):
        hri = object.__new__(HRI_Agent)
        hri.agent = FakeStreamingAgent(
            parts,
            {"messages": [final_message], "llm_calls": 1},
        )
        hri.print_raw = print_raw
        return hri

    def test_reasoning_and_response_use_separate_sections(self) -> None:
        complete = AIMessage(
            content="final answer",
            additional_kwargs={"reasoning": "reasoning text"},
        )
        hri = self.make_hri(
            [
                message_part(
                    AIMessageChunk(
                        content="",
                        additional_kwargs={"reasoning": "reasoning text"},
                    )
                ),
                message_part(AIMessageChunk(content="final answer")),
                update_part("HRI Agent", [complete]),
            ],
            complete,
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = hri.invoke_agent([], current_frame={})

        rendered = output.getvalue()
        self.assertLess(
            rendered.index("HRI Agent Thinking:"),
            rendered.index("reasoning text"),
        )
        self.assertLess(
            rendered.index("reasoning text"),
            rendered.index("HRI Agent Response:"),
        )
        self.assertLess(
            rendered.index("HRI Agent Response:"),
            rendered.index("final answer"),
        )
        self.assertEqual(rendered.count("final answer"), 1)
        self.assertNotIn("reasoning textfinal answer", rendered)
        self.assertNotIn("[graph update]", rendered)
        self.assertIs(result["messages"][-1], complete)

    def test_print_raw_outputs_complete_model_message_as_json(self) -> None:
        complete = AIMessage(
            content="final answer",
            additional_kwargs={"reasoning": "reasoning text"},
            response_metadata={"finish_reason": "stop"},
        )
        hri = self.make_hri(
            [
                message_part(AIMessageChunk(content="final answer")),
                update_part("HRI Agent", [complete]),
            ],
            complete,
            print_raw=True,
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hri.invoke_agent([], current_frame={})

        rendered = output.getvalue()
        self.assertIn("HRI Agent Raw Response #1:", rendered)
        self.assertIn('"content": "final answer"', rendered)
        self.assertIn('"reasoning": "reasoning text"', rendered)
        self.assertIn('"finish_reason": "stop"', rendered)

    def test_tool_call_deltas_and_results_have_separate_sections(self) -> None:
        complete = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "call_sub_agent",
                    "args": {"sub_agent_name": "Planner"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        result = ToolMessage(
            content="Finished planning.",
            name="call_sub_agent",
            tool_call_id="call-1",
        )
        hri = self.make_hri(
            [
                message_part(
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            tool_call_chunk(
                                name="call_sub_agent",
                                args='{"sub_agent_name":"Planner"}',
                                id="call-1",
                                index=0,
                            )
                        ],
                    )
                ),
                update_part("HRI Agent", [complete]),
                update_part("Tool Node", [result]),
            ],
            complete,
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hri.invoke_agent([], current_frame={})

        rendered = output.getvalue()
        self.assertIn("HRI Agent Tool Call:", rendered)
        self.assertIn("call_sub_agent arguments=", rendered)
        self.assertIn('{"sub_agent_name":"Planner"}', rendered)
        self.assertIn("HRI Agent Tool Result:", rendered)
        self.assertIn("call_sub_agent: Finished planning.", rendered)


if __name__ == "__main__":
    unittest.main()
