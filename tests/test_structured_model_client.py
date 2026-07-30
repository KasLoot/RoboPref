from __future__ import annotations

import unittest

from pydantic import BaseModel

from prefmem.agents.model_client import (
    ModelEndpoint,
    StructuredAgentClient,
    StructuredModelError,
)


class Answer(BaseModel):
    value: str


class FakeRunnable:
    def __init__(self, result) -> None:
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class FakeModel:
    def __init__(self, result) -> None:
        self.runnable = FakeRunnable(result)
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return self.runnable


class StructuredAgentClientTests(unittest.TestCase):
    def test_schema_and_visual_context_are_sent_together(self) -> None:
        model = FakeModel(
            {
                "parsed": Answer(value="ok"),
                "raw": {"content": '{"value":"ok"}'},
                "parsing_error": None,
            }
        )
        client = StructuredAgentClient(
            ModelEndpoint(
                model="model",
                base_url="http://127.0.0.1:8000/v1",
            ),
            model=model,
        )

        answer = client.invoke(
            agent="test",
            schema=Answer,
            system_prompt="Return an answer.",
            text="Inspect the observations.",
            visual_blocks=[
                {"type": "text", "text": "First frame:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,frame"},
                },
            ],
        )

        self.assertEqual(answer.value, "ok")
        self.assertEqual(model.calls[0][0], Answer)
        self.assertEqual(model.calls[0][1]["method"], "json_schema")
        human_content = model.runnable.messages[-1].content
        self.assertEqual(human_content[0]["text"], "First frame:")
        self.assertEqual(human_content[-1]["text"], "Inspect the observations.")

    def test_parsing_failure_is_an_explicit_boundary_error(self) -> None:
        model = FakeModel(
            {
                "parsed": None,
                "raw": {"content": "not JSON"},
                "parsing_error": "invalid",
            }
        )
        client = StructuredAgentClient(
            ModelEndpoint(
                model="model",
                base_url="http://127.0.0.1:8000/v1",
            ),
            model=model,
        )

        with self.assertRaisesRegex(StructuredModelError, "valid Answer"):
            client.invoke(
                agent="test",
                schema=Answer,
                system_prompt="Return an answer.",
                text="hello",
            )


if __name__ == "__main__":
    unittest.main()
