from __future__ import annotations

from types import SimpleNamespace
import unittest

from prefmem.memory.embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbeddingDocument,
    EmbeddingGemmaClient,
    EmbeddingModelDiscoveryError,
    EmbeddingProtocolError,
    cosine_similarity,
)


def _vector(size: int, marker: float) -> list[float]:
    values = [0.0] * size
    values[0] = marker
    return values


class _FakeModels:
    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = model_ids
        self.calls = 0

    def list(self):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


class _FakeEmbeddings:
    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        inputs = request["input"]
        # Reverse response order to verify that response indices restore it.
        data = [
            SimpleNamespace(
                index=index,
                embedding=_vector(self.dimensions, float(index + 1)),
            )
            for index in reversed(range(len(inputs)))
        ]
        return SimpleNamespace(data=data, model=request["model"])


class _FakeClient:
    def __init__(
        self,
        *,
        model_ids: list[str] | None = None,
        dimensions: int = 768,
    ) -> None:
        self.models = _FakeModels(model_ids or ["embeddinggemma-300m"])
        self.embeddings = _FakeEmbeddings(dimensions)


class EmbeddingGemmaClientTests(unittest.TestCase):
    def test_query_uses_retrieval_query_prefix_and_discovers_model(self) -> None:
        fake = _FakeClient()
        client = EmbeddingGemmaClient(client=fake)

        embedding = client.embed_query("tidy the table")

        self.assertEqual(len(embedding), 768)
        self.assertEqual(fake.models.calls, 1)
        self.assertEqual(client.resolved_model, "embeddinggemma-300m")
        request = fake.embeddings.requests[0]
        self.assertEqual(
            request["input"],
            [f"{QUERY_PREFIX}tidy the table"],
        )
        self.assertEqual(request["encoding_format"], "float")
        # Native output does not need a Matryoshka request parameter.
        self.assertNotIn("dimensions", request)

    def test_documents_use_titles_batch_and_preserve_input_order(self) -> None:
        fake = _FakeClient()
        client = EmbeddingGemmaClient(client=fake, max_batch_size=2)
        documents = [
            EmbeddingDocument("Books go left.", title="Table layout"),
            EmbeddingDocument("Electronics go right."),
            EmbeddingDocument("Use the blue mug.", title=""),
        ]

        vectors = client.embed_documents(documents)

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(fake.embeddings.requests), 2)
        self.assertEqual(
            fake.embeddings.requests[0]["input"],
            [
                f"{DOCUMENT_PREFIX.format(title='Table layout')}Books go left.",
                f"{DOCUMENT_PREFIX.format(title='none')}Electronics go right.",
            ],
        )
        self.assertEqual(
            fake.embeddings.requests[1]["input"],
            [f"{DOCUMENT_PREFIX.format(title='none')}Use the blue mug."],
        )
        # The fake reverses each response batch; index sorting restores order.
        self.assertEqual(vectors[0][0], 1.0)
        self.assertEqual(vectors[1][0], 2.0)
        self.assertEqual(vectors[2][0], 1.0)
        self.assertEqual(fake.models.calls, 1)

    def test_reduced_dimensions_are_sent_and_validated(self) -> None:
        fake = _FakeClient(dimensions=256)
        client = EmbeddingGemmaClient(client=fake, dimensions=256)

        vector = client.embed_document("A compact memory.")

        self.assertEqual(len(vector), 256)
        self.assertEqual(fake.embeddings.requests[0]["dimensions"], 256)

    def test_configured_model_must_be_advertised(self) -> None:
        fake = _FakeClient(model_ids=["served-name"])
        client = EmbeddingGemmaClient(
            client=fake,
            model="/data/models/embeddinggemma-300m",
        )

        with self.assertRaisesRegex(
            EmbeddingModelDiscoveryError,
            "available models: served-name",
        ):
            client.embed_query("hello")
        self.assertEqual(fake.embeddings.requests, [])

    def test_ambiguous_discovery_requires_configured_model(self) -> None:
        fake = _FakeClient(model_ids=["model-a", "model-b"])
        client = EmbeddingGemmaClient(client=fake)

        with self.assertRaisesRegex(
            EmbeddingModelDiscoveryError,
            "multiple model IDs",
        ):
            client.discover_model()

    def test_response_dimension_mismatch_is_rejected(self) -> None:
        fake = _FakeClient(dimensions=512)
        client = EmbeddingGemmaClient(client=fake, dimensions=768)

        with self.assertRaisesRegex(
            EmbeddingProtocolError,
            "has 512 dimensions; expected 768",
        ):
            client.embed_query("hello")

    def test_duplicate_response_indices_are_rejected(self) -> None:
        fake = _FakeClient()

        def duplicate_indices(**request):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=_vector(768, 1.0)),
                    SimpleNamespace(index=0, embedding=_vector(768, 2.0)),
                ]
            )

        fake.embeddings.create = duplicate_indices
        client = EmbeddingGemmaClient(client=fake)

        with self.assertRaisesRegex(EmbeddingProtocolError, "repeats index 0"):
            client.embed_documents(
                [EmbeddingDocument("one"), EmbeddingDocument("two")]
            )

    def test_empty_inputs_are_rejected_without_server_call(self) -> None:
        fake = _FakeClient()
        client = EmbeddingGemmaClient(client=fake)

        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            client.embed_query("  ")
        self.assertEqual(fake.models.calls, 0)
        self.assertEqual(client.embed_documents([]), [])

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "128, 256, 512, 768"):
            EmbeddingGemmaClient(client=_FakeClient(), dimensions=384)


class CosineSimilarityTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_cosine_rejects_invalid_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            cosine_similarity([1], [1, 2])
        with self.assertRaisesRegex(ValueError, "zero vector"):
            cosine_similarity([0, 0], [1, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            cosine_similarity([float("nan")], [1])


if __name__ == "__main__":
    unittest.main()
