from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from prefmem.agents.memory import Memory_Agent


class StubEmbeddingModel:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode_query(self, texts: list[str]):
        return [self.vectors[text] for text in texts]


def make_agent() -> Memory_Agent:
    agent = object.__new__(Memory_Agent)
    agent.pref_json = [
        {"id": "pref-a", "text": "Alpha preference"},
        {"id": "pref-b", "text": "Bridge preference"},
        {"id": "pref-c", "text": "Charlie preference"},
    ]
    agent.pref_embedding = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
        dtype=np.float32,
    )
    agent.embedding_model = StubEmbeddingModel(
        {"alpha query": [1.0, 0.0], "charlie query": [0.0, 1.0]}
    )
    agent.top_k = 2
    agent.top_cap_k = 2
    agent.args = SimpleNamespace(memory_store_path=None)
    agent._trace_sink = None
    agent._reset_request_state("RETRIEVE")
    agent._original_request = "RETRIEVE REQUEST: test preferences"
    return agent


class MemoryTraceTests(unittest.TestCase):
    def test_trace_records_rank_merge_deduplication_and_cap(self) -> None:
        agent = make_agent()
        snapshots: list[dict] = []
        agent.set_trace_sink(snapshots.append)

        results = agent.retrieve(["alpha query", "charlie query"])

        self.assertEqual([item["id"] for item in results], ["pref-a", "pref-c"])
        trace = agent.last_retrieval_trace
        assert trace is not None
        self.assertEqual(trace["query_count"], 2)
        self.assertEqual(trace["n_current"], 3)
        self.assertEqual(trace["top_k_effective"], 2)
        self.assertEqual(len(trace["per_query_candidates"]), 4)
        self.assertEqual(len(trace["merged_before_cap"]), 3)
        self.assertEqual(len(trace["capped_candidates"]), 2)
        self.assertEqual(trace["stage"], "RETRIEVAL_CAPPED")
        self.assertEqual(len(snapshots), 1)

    def test_semantic_finalization_validates_provenance_and_updates_trace(self) -> None:
        agent = make_agent()
        results = agent.retrieve(["alpha query", "charlie query"])
        returned = results[0]

        agent._finalize_semantic_trace(
            json.dumps(
                {
                    "status": "FOUND",
                    "retrieved_memory": [returned],
                    "warnings": [],
                }
            )
        )

        trace = agent.last_retrieval_trace
        assert trace is not None
        self.assertEqual(trace["stage"], "SEMANTIC_FILTER_COMPLETE")
        self.assertEqual(trace["final_memory_ids"], [returned["id"]])
        decisions = {
            item["memory_id"]: item for item in trace["semantic_decisions"]
        }
        self.assertTrue(decisions[returned["id"]]["returned"])
        self.assertEqual(decisions[returned["id"]]["label"], "APPLICABLE_CONTEXT")

    def test_semantic_finalization_rejects_fabricated_candidate(self) -> None:
        agent = make_agent()
        agent.retrieve(["alpha query", "charlie query"])

        with self.assertRaisesRegex(ValueError, "not from the capped"):
            agent._finalize_semantic_trace(
                json.dumps(
                    {
                        "status": "FOUND",
                        "retrieved_memory": [
                            {
                                "id": "pref-fabricated",
                                "memory_type": "PREFERENCE",
                                "text": "Invented",
                                "similarity": 1.0,
                            }
                        ],
                        "warnings": [],
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
