from __future__ import annotations

import unittest
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage

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


if __name__ == "__main__":
    unittest.main()
