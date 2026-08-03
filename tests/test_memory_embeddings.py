from __future__ import annotations

import unittest
from unittest.mock import patch

from prefmem.agents.memory import VLLMEmbeddingGemma


class RecordingEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class VLLMEmbeddingGemmaTests(unittest.TestCase):
    def test_configures_openai_compatible_embedding_client(self) -> None:
        with patch("prefmem.agents.memory.OpenAIEmbeddings") as client_class:
            VLLMEmbeddingGemma(
                model="/models/embeddinggemma-300m",
                base_url="http://localhost:8080/v1",
            )

        client_class.assert_called_once_with(
            model="/models/embeddinggemma-300m",
            api_key="EMPTY",
            base_url="http://localhost:8080/v1",
            check_embedding_ctx_length=False,
        )

    def test_document_embeddings_use_document_prompt(self) -> None:
        adapter = object.__new__(VLLMEmbeddingGemma)
        adapter._client = RecordingEmbeddings()

        result = adapter.encode_document(["first memory", "second memory"])

        self.assertEqual(
            adapter._client.calls,
            [[
                "title: none | text: first memory",
                "title: none | text: second memory",
            ]],
        )
        self.assertEqual(result, [[0.0, 1.0], [1.0, 1.0]])

    def test_query_embeddings_are_batched_with_query_prompt(self) -> None:
        adapter = object.__new__(VLLMEmbeddingGemma)
        adapter._client = RecordingEmbeddings()

        result = adapter.encode_query(["first query", "second query"])

        self.assertEqual(
            adapter._client.calls,
            [[
                "task: search result | query: first query",
                "task: search result | query: second query",
            ]],
        )
        self.assertEqual(result, [[0.0, 1.0], [1.0, 1.0]])

    def test_single_query_preserves_sentence_transformer_shape(self) -> None:
        adapter = object.__new__(VLLMEmbeddingGemma)
        adapter._client = RecordingEmbeddings()

        result = adapter.encode_query("one query")

        self.assertEqual(result, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
