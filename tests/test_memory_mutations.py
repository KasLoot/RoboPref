from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np
from langchain.messages import AIMessage

from prefmem.agents.memory import Memory_Agent


class StubEmbeddingModel:
    def __init__(
        self,
        *,
        query_vectors: dict[str, list[float]] | None = None,
        document_vectors: dict[str, list[float]] | None = None,
    ) -> None:
        self.query_vectors = query_vectors or {}
        self.document_vectors = document_vectors or {}

    def encode_query(self, texts: str | list[str]):
        if isinstance(texts, str):
            return self.query_vectors[texts]
        return [self.query_vectors[text] for text in texts]

    def encode_document(self, texts: list[str]):
        return [self.document_vectors[text] for text in texts]


def make_agent(
    memories: list[dict],
    embeddings: list[list[float]],
    embedding_model: StubEmbeddingModel,
    *,
    request_type: str = "MUTATE",
) -> Memory_Agent:
    agent = object.__new__(Memory_Agent)
    agent.pref_json = [memory.copy() for memory in memories]
    agent.pref_embedding = np.asarray(embeddings, dtype=np.float32)
    agent.embedding_model = embedding_model
    agent.top_k = 5
    agent.top_cap_k = 10
    agent.args = SimpleNamespace(memory_store_path=None)
    agent._reset_request_state(request_type)
    return agent


class MemoryMutationTests(unittest.TestCase):
    def test_retrieve_returns_memory_ids(self) -> None:
        agent = make_agent(
            memories=[{"id": "pref-one", "text": "Clear the table."}],
            embeddings=[[1.0, 0.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"table preference": [1.0, 0.0]},
            ),
            request_type="RETRIEVE",
        )

        result = agent.retrieve(["table preference"])

        self.assertEqual(result[0]["id"], "pref-one")
        self.assertEqual(result[0]["text"], "Clear the table.")

    def test_update_uses_id_from_current_retrieval(self) -> None:
        new_text = (
            "When the user says 'tidy up', clear the table and clean it."
        )
        agent = make_agent(
            memories=[
                {
                    "id": "pref-tidy",
                    "text": "When the user says 'tidy up', clear the table.",
                },
                {
                    "id": "pref-cup",
                    "text": "Put cups on the upper shelf.",
                },
            ],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"the saved meaning of tidy up": [1.0, 0.0]},
                document_vectors={new_text: [0.8, 0.6]},
            ),
        )
        agent.retrieve(["the saved meaning of tidy up"])

        result = agent.update("pref-tidy", new_text)

        self.assertEqual(result["status"], "UPDATED")
        self.assertEqual(result["id"], "pref-tidy")
        self.assertEqual(
            result["previous_text"],
            "When the user says 'tidy up', clear the table.",
        )
        self.assertEqual(agent.pref_json[0]["text"], new_text)
        self.assertEqual(
            agent.pref_json[1]["text"],
            "Put cups on the upper shelf.",
        )

    def test_forget_uses_id_from_current_retrieval(self) -> None:
        agent = make_agent(
            memories=[
                {"id": "pref-tidy", "text": "Clear the table."},
                {"id": "pref-cup", "text": "Put cups on the shelf."},
            ],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"cup preference": [0.0, 1.0]},
            ),
        )
        agent.retrieve(["cup preference"])

        result = agent.forget("pref-cup")

        self.assertEqual(result["status"], "FORGOTTEN")
        self.assertEqual(result["id"], "pref-cup")
        self.assertEqual(
            agent.pref_json,
            [{"id": "pref-tidy", "text": "Clear the table."}],
        )

    def test_mutation_before_retrieval_is_rejected(self) -> None:
        agent = make_agent(
            memories=[{"id": "pref-one", "text": "Clear the table."}],
            embeddings=[[1.0, 0.0]],
            embedding_model=StubEmbeddingModel(),
        )

        result = agent.forget("pref-one")

        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(len(agent.pref_json), 1)

    def test_id_not_returned_by_retrieval_is_rejected(self) -> None:
        agent = make_agent(
            memories=[{"id": "pref-one", "text": "Clear the table."}],
            embeddings=[[1.0, 0.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"table preference": [1.0, 0.0]},
                document_vectors={"replacement": [0.0, 1.0]},
            ),
        )
        agent.retrieve(["table preference"])

        result = agent.update("placeholder-id", "replacement")

        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(agent.pref_json[0]["text"], "Clear the table.")

    def test_retrieve_request_cannot_write(self) -> None:
        agent = make_agent(
            memories=[{"id": "pref-one", "text": "Clear the table."}],
            embeddings=[[1.0, 0.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"table preference": [1.0, 0.0]},
            ),
            request_type="RETRIEVE",
        )
        agent.retrieve(["table preference"])

        result = agent.forget("pref-one")

        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(len(agent.pref_json), 1)

    def test_identical_remember_request_is_unchanged(self) -> None:
        text = "When the user says tidy up, clear and clean the table."
        agent = make_agent(
            memories=[{"id": "pref-tidy", "text": text}],
            embeddings=[[1.0, 0.0]],
            embedding_model=StubEmbeddingModel(
                query_vectors={"tidy up preference": [1.0, 0.0]},
            ),
        )
        agent.retrieve(["tidy up preference"])

        result = agent.remember(text.upper())

        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["id"], "pref-tidy")
        self.assertEqual(len(agent.pref_json), 1)

    def test_new_memory_can_be_remembered_after_empty_retrieval(self) -> None:
        text = "Put clean cups on the upper shelf."
        agent = make_agent(
            memories=[],
            embeddings=[],
            embedding_model=StubEmbeddingModel(
                document_vectors={text: [1.0, 0.0]},
            ),
        )
        agent.retrieve(["cup storage preference"])

        result = agent.remember(text)

        self.assertEqual(result["status"], "REMEMBERED")
        self.assertEqual(len(agent.pref_json), 1)

    def test_retrieve_can_only_run_once_per_request(self) -> None:
        agent = make_agent(
            memories=[],
            embeddings=[],
            embedding_model=StubEmbeddingModel(),
        )
        agent.retrieve(["first query"])

        with self.assertRaisesRegex(ValueError, "only be called once"):
            agent.retrieve(["second query"])

    def test_tool_failure_becomes_error_message(self) -> None:
        class FailingTool:
            def invoke(self, _args):
                raise ValueError("bad tool input")

        agent = object.__new__(Memory_Agent)
        agent.TOOLS_BY_NAME = {"failing": FailingTool()}
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "failing", "args": {}, "id": "call-1"}
                    ],
                )
            ]
        }

        result = agent.tool_node(state)
        content = json.loads(result["messages"][0].content)

        self.assertEqual(content["status"], "ERROR")
        self.assertEqual(content["tool"], "failing")
        self.assertEqual(content["message"], "bad tool input")

    def test_retrieval_list_is_serialized_as_json_text(self) -> None:
        class RetrievalTool:
            def invoke(self, _args):
                return [
                    {
                        "id": "pref-one",
                        "memory_type": "PREFERENCE",
                        "text": "Clear the table.",
                        "similarity": 0.9,
                    }
                ]

        agent = object.__new__(Memory_Agent)
        agent.TOOLS_BY_NAME = {"retrieve": RetrievalTool()}
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "retrieve", "args": {}, "id": "call-1"}
                    ],
                )
            ]
        }

        result = agent.tool_node(state)
        content = result["messages"][0].content

        self.assertIsInstance(content, str)
        self.assertEqual(json.loads(content)[0]["id"], "pref-one")


if __name__ == "__main__":
    unittest.main()
