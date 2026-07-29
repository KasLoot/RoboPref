from __future__ import annotations

import unittest
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from prefmem.agents.hri import HRI_Agent


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content="ok")


class HRIImageHistoryTests(unittest.TestCase):
    def test_only_current_turn_image_is_sent_to_model(self) -> None:
        model = RecordingModel()
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="test system prompt")
        hri.args = SimpleNamespace(print_raw=False)
        hri.hri_llm = model
        hri.metrics = None
        hri.TOOLS_BY_NAME = {}
        hri.hri_agent = hri.build_agent()

        first_frame = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,first"},
        }
        second_frame = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,second"},
        }

        hri.invoke_agent(
            [HumanMessage(content="first question")],
            current_frame=first_frame,
        )
        result = hri.invoke_agent(
            [HumanMessage(content="second question")],
            current_frame=second_frame,
        )

        second_model_call = model.calls[-1]
        human_messages = [
            message
            for message in second_model_call
            if isinstance(message, HumanMessage)
        ]

        self.assertEqual(human_messages[0].content, "first question")
        self.assertEqual(
            human_messages[1].content,
            [
                second_frame,
                {"type": "text", "text": "second question"},
            ],
        )

        image_blocks = [
            block
            for message in human_messages
            if isinstance(message.content, list)
            for block in message.content
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
        self.assertEqual(image_blocks, [second_frame])

        checkpointed_human_messages = [
            message
            for message in result["messages"]
            if isinstance(message, HumanMessage)
        ]
        self.assertEqual(
            [message.content for message in checkpointed_human_messages],
            ["first question", "second question"],
        )

    def test_prior_turn_reasoning_and_tool_trace_are_not_sent(self) -> None:
        model = RecordingModel()
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="test system prompt")
        hri.hri_llm = model
        hri.metrics = None
        hri.thinking_enabled = False

        prior_tool_call = AIMessage(
            content="",
            additional_kwargs={"reasoning": "private tool reasoning"},
            tool_calls=[
                {
                    "name": "call_sub_agent",
                    "args": {"message": "make a plan"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        prior_tool_result = ToolMessage(
            content="private tool result",
            tool_call_id="call-1",
        )
        prior_response = AIMessage(
            content="public answer",
            additional_kwargs={"reasoning": "private final reasoning"},
        )

        hri.llm_call(
            {
                "messages": [
                    HumanMessage(content="first question"),
                    prior_tool_call,
                    prior_tool_result,
                    prior_response,
                    HumanMessage(content="second question"),
                ],
                "llm_calls": 1,
            },
            SimpleNamespace(context={"current_frame": {}}),
        )

        sent_messages = model.calls[-1]
        self.assertEqual(
            [
                message.content
                for message in sent_messages
                if isinstance(message, HumanMessage)
            ],
            [
                "first question",
                [
                    {},
                    {"type": "text", "text": "second question"},
                ],
            ],
        )
        sent_ai_messages = [
            message
            for message in sent_messages
            if isinstance(message, AIMessage)
        ]
        self.assertEqual(
            [message.content for message in sent_ai_messages],
            ["public answer"],
        )
        self.assertNotIn(
            "reasoning",
            sent_ai_messages[0].additional_kwargs,
        )
        self.assertFalse(
            any(
                isinstance(message, ToolMessage)
                for message in sent_messages
            )
        )

    def test_current_turn_tool_trace_is_sent_to_follow_up_call(self) -> None:
        model = RecordingModel()
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="test system prompt")
        hri.hri_llm = model
        hri.metrics = None
        hri.thinking_enabled = False

        current_tool_call = AIMessage(
            content="",
            additional_kwargs={"reasoning": "current reasoning"},
            tool_calls=[
                {
                    "name": "call_sub_agent",
                    "args": {"message": "make a plan"},
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        )
        current_tool_result = ToolMessage(
            content="current tool result",
            tool_call_id="call-2",
        )

        hri.llm_call(
            {
                "messages": [
                    HumanMessage(content="first question"),
                    AIMessage(content="first answer"),
                    HumanMessage(content="second question"),
                    current_tool_call,
                    current_tool_result,
                ],
                "llm_calls": 2,
            },
            SimpleNamespace(context={"current_frame": {}}),
        )

        sent_messages = model.calls[-1]
        self.assertIn(current_tool_call, sent_messages)
        self.assertIn(current_tool_result, sent_messages)


if __name__ == "__main__":
    unittest.main()
