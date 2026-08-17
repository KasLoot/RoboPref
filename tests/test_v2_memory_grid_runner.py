from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments_suite_v2.cases.memory_grid import all_memory_grid_needs
from experiments_suite_v2.runners.memory_grid import (
    EmbeddedMemoryGrid,
    MemoryGridRow,
    build_memory_grid_rows,
    evaluate_memory_grid_row,
)


class MemoryGridRunnerTests(unittest.TestCase):
    def test_row_expansion_is_complete_unique_and_deterministic(self) -> None:
        first = build_memory_grid_rows()
        second = build_memory_grid_rows()
        self.assertEqual(len(first), 19_200)
        self.assertEqual([row.row_id for row in first], [row.row_id for row in second])
        self.assertEqual(len({row.row_id for row in first}), 19_200)
        cells = {
            (row.query_count, row.top_k, row.top_cap_k, row.store_size)
            for row in first
        }
        self.assertEqual(len(cells), 240)

    def test_evaluation_uses_production_merge_and_cap(self) -> None:
        need = all_memory_grid_needs()[0]
        dimension = 4
        document_vectors = {
            document.text: np.array(
                [1.0, index / 100.0, 0.0, 0.0], dtype=np.float32
            )
            for index, document in enumerate(need.documents)
        }
        query_vectors = {
            query: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for query in need.queries
        }
        embeddings = EmbeddedMemoryGrid(
            document_vectors_by_text=document_vectors,
            query_vectors_by_text=query_vectors,
            model="fixture",
            base_url="fixture://embedding",
        )
        self.assertEqual(embeddings.dimension, dimension)
        row = MemoryGridRow(
            row_id="fixture-row",
            order_index=0,
            context_id=need.context_id,
            split=need.split,
            need_type=need.need_type,
            query_count=3,
            top_k=3,
            top_cap_k=2,
            store_size=5,
        )
        evaluation = evaluate_memory_grid_row(
            row,
            need=need,
            embeddings=embeddings,
        )
        self.assertEqual(len(evaluation.returned_ids), 2)
        self.assertEqual(
            evaluation.result["production_retrieval_method"],
            "prefmem.agents.memory.Memory_Agent._retrieve",
        )
        self.assertEqual(evaluation.trace["query_count"], 3)
        self.assertEqual(evaluation.trace["top_k_configured"], 3)
        self.assertEqual(evaluation.trace["top_cap_k"], 2)

    def test_embedding_cache_round_trip_is_hash_verified(self) -> None:
        need = all_memory_grid_needs()[0]
        document_vectors = {
            document.text: np.array([1.0, index + 1.0], dtype=np.float32)
            for index, document in enumerate(need.documents[:5])
        }
        query_vectors = {
            query: np.array([2.0, index + 1.0], dtype=np.float32)
            for index, query in enumerate(need.queries)
        }
        cache = EmbeddedMemoryGrid(
            document_vectors_by_text=document_vectors,
            query_vectors_by_text=query_vectors,
            model="fixture-model",
            base_url="http://fixture.invalid/v1",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = cache.save(directory)
            restored = EmbeddedMemoryGrid.load(directory)
        self.assertEqual(restored.cache_sha256, cache.cache_sha256)
        self.assertEqual(metadata["cache_sha256"], cache.cache_sha256)
        self.assertEqual(restored.dimension, 2)


if __name__ == "__main__":
    unittest.main()
