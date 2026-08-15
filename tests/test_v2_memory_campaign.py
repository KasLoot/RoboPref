from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments_suite_v2.cases.memory_grid import all_memory_grid_needs
from experiments_suite_v2.io import iter_jsonl, load_json
from experiments_suite_v2.runners.memory_campaign import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEVELOPMENT_SELECTION_RULE_SHA256,
    EXPECTED_CONFIRMATORY_NEEDS,
    EXPECTED_DEVELOPMENT_NEEDS,
    EXPECTED_DOCUMENT_TEXT_COUNT,
    EXPECTED_PARAMETER_CELLS,
    EXPECTED_QUERY_TEXT_COUNT,
    EXPECTED_RESULT_ROWS,
    MemoryCampaign,
    MemoryCampaignEndpointInvalid,
    MemoryCampaignError,
    first_q1k1_recovery_counterexample,
    frozen_embedding_texts,
    select_development_configuration,
)
from experiments_suite_v2.runners.memory_grid import build_memory_grid_rows
from experiments_suite_v2.storage import RunStore
from prefmem.agents.memory import Memory_Agent


SOURCE_COMMIT = "memory-campaign-test"


def _vector(text: str, dimension: int = 8) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.asarray([digest[index] + 1 for index in range(dimension)], dtype=np.float32)


class _DeterministicEmbeddingAdapter:
    def __init__(self, before_call=None) -> None:
        self.before_call = before_call
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def _encode(self, kind: str, texts):
        values = tuple(texts)
        if self.before_call is not None:
            self.before_call(kind, values)
        self.calls.append((kind, values))
        return [_vector(text) for text in values]

    def encode_document(self, texts):
        return self._encode("DOCUMENT", texts)

    def encode_query(self, texts):
        return self._encode("QUERY", texts)


class _DisconnectedAdapter:
    def encode_document(self, _texts):
        raise ConnectionError("SSH-forwarded EmbeddingGemma endpoint disconnected")

    def encode_query(self, _texts):
        raise AssertionError("query call must not follow document outage")


class _QueryDisconnectedAfterDocuments(_DeterministicEmbeddingAdapter):
    def encode_query(self, _texts):
        raise ConnectionError("endpoint disconnected after cached document batch")


class _Factory:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, base_url: str):
        self.calls.append((model, base_url))
        return self.adapter


def _scored_row(
    *,
    context_id: str,
    configuration: tuple[int, int, int, int],
    recall: float,
    mrr: float,
    precision: float,
    required=("required",),
    returned=("required",),
    split="DEVELOPMENT",
):
    q, k, cap, size = configuration
    return {
        "row_id": f"{context_id}-q{q}k{k}c{cap}n{size}",
        "context_id": context_id,
        "split": split,
        "need_type": "ONE_RELEVANT",
        "query_count": q,
        "top_k": k,
        "top_cap_k": cap,
        "store_size": size,
        "required_memory_ids": list(required),
        "returned_ids": list(returned),
        "metrics": {
            "required_memory_recall_at_cap": recall,
            "required_memory_mrr": mrr,
            "candidate_precision": precision,
            "harmful_candidate_count": 0,
            "context_utf8_bytes": len(returned) * 10,
            "returned_candidate_count": len(returned),
            "latency_seconds": 0.01,
        },
    }


class MemoryCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.suite_root = Path(self.temporary.name)
        self.store = RunStore(self.suite_root)
        self.store.create_run(
            "memory-run",
            source_commit=SOURCE_COMMIT,
            dirty_state={"dirty": False},
            runtime_options={"campaign": "AB-MEM-Q"},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _subset(self, *, row_count=1, factory=None):
        needs = all_memory_grid_needs()[:1]
        rows = build_memory_grid_rows(needs)[:row_count]
        adapter_factory = factory or _Factory(_DeterministicEmbeddingAdapter())
        return MemoryCampaign(
            store=self.store,
            suite_run_id="memory-run",
            embedding_adapter_factory=adapter_factory,
            _test_needs=needs,
            _test_rows=rows,
        )

    def test_manifest_first_freezes_exact_19200_rows_without_live_call(self):
        factory = _Factory(_DeterministicEmbeddingAdapter())
        campaign = MemoryCampaign(
            store=self.store,
            suite_run_id="memory-run",
            embedding_adapter_factory=factory,
        )
        manifest = load_json(campaign.attempt_dir / "attempt_manifest.json")
        self.assertEqual(manifest["artifact_profile"], "BATCH_EVAL")
        self.assertEqual(manifest["status"], "RUNNING")
        self.assertEqual(manifest["selected_services"][0]["service_id"], "embeddinggemma-retriever-v1")
        self.assertEqual(factory.calls, [])
        self.assertFalse((campaign.attempt_dir / "model_calls.jsonl").exists())

        count = 0
        row_ids = set()
        first = last = None
        for record in iter_jsonl(campaign.attempt_dir / "input_rows.jsonl"):
            first = record if first is None else first
            last = record
            count += 1
            row_ids.add(record["row_id"])
            self.assertEqual(record["identity"]["attempt_number"], 0)
            self.assertEqual(record["trial_tuple"]["artifact_profile"], "BATCH_EVAL")
        self.assertEqual(count, EXPECTED_RESULT_ROWS)
        self.assertEqual(len(row_ids), EXPECTED_RESULT_ROWS)
        self.assertEqual(first["order_index"], 0)
        self.assertEqual(last["order_index"], EXPECTED_RESULT_ROWS - 1)
        self.assertEqual(len(campaign.document_texts), EXPECTED_DOCUMENT_TEXT_COUNT)
        self.assertEqual(len(campaign.query_texts), EXPECTED_QUERY_TEXT_COUNT)
        self.assertEqual(
            len(
                {
                    (row.query_count, row.top_k, row.top_cap_k, row.store_size)
                    for row in campaign.rows
                }
            ),
            EXPECTED_PARAMETER_CELLS,
        )

    def test_embedding_inventory_and_calls_are_prejournaled_then_hash_cached(self):
        documents, queries = frozen_embedding_texts()
        self.assertEqual(len(documents), EXPECTED_DOCUMENT_TEXT_COUNT)
        self.assertEqual(len(queries), EXPECTED_QUERY_TEXT_COUNT)
        observations = []

        campaign_box = {}

        def before_call(kind, texts):
            campaign = campaign_box["campaign"]
            self.assertTrue((campaign.attempt_dir / "attempt_manifest.json").is_file())
            self.assertTrue((campaign.attempt_dir / "input_rows.jsonl").is_file())
            journal = list(iter_jsonl(campaign.attempt_dir / "model_calls.jsonl"))
            self.assertEqual(journal[-1]["event_type"], "EMBEDDING_CALL_ALLOCATED")
            self.assertEqual(journal[-1]["kind"], kind)
            self.assertEqual(journal[-1]["input_texts"], list(texts))
            observations.append(journal[-1]["call_attempt_id"])

        adapter = _DeterministicEmbeddingAdapter(before_call)
        factory = _Factory(adapter)
        rows = build_memory_grid_rows()[:1]
        campaign = MemoryCampaign(
            store=self.store,
            suite_run_id="memory-run",
            embedding_adapter_factory=factory,
            _test_rows=rows,
        )
        campaign_box["campaign"] = campaign
        cache = campaign.ensure_embedding_cache()
        self.assertEqual(len(factory.calls), 1)
        expected_batches = math.ceil(EXPECTED_DOCUMENT_TEXT_COUNT / DEFAULT_EMBEDDING_BATCH_SIZE) + math.ceil(
            EXPECTED_QUERY_TEXT_COUNT / DEFAULT_EMBEDDING_BATCH_SIZE
        )
        self.assertEqual(len(adapter.calls), expected_batches)
        self.assertEqual(len(observations), expected_batches)
        journal = list(iter_jsonl(campaign.attempt_dir / "model_calls.jsonl"))
        self.assertEqual(len(journal), expected_batches * 2)
        self.assertEqual(
            {row["event_type"] for row in journal},
            {"EMBEDDING_CALL_ALLOCATED", "EMBEDDING_CALL_SUCCEEDED"},
        )
        restored = campaign._load_final_cache()
        self.assertIsNotNone(restored)
        self.assertEqual(restored.cache_sha256, cache.cache_sha256)
        self.assertEqual(len(restored.document_vectors_by_text), EXPECTED_DOCUMENT_TEXT_COUNT)
        self.assertEqual(len(restored.query_vectors_by_text), EXPECTED_QUERY_TEXT_COUNT)

    def test_endpoint_invalid_call_gets_exact_a1_retry_without_row_duplication(self):
        failed_factory = _Factory(_DisconnectedAdapter())
        campaign = self._subset(factory=failed_factory)
        with self.assertRaises(MemoryCampaignEndpointInvalid) as caught:
            campaign.ensure_embedding_cache()
        self.assertTrue(caught.exception.call_attempt_id.endswith(".A0"))
        self.assertEqual(campaign.progress().result_rows, 0)
        input_before = (campaign.attempt_dir / "input_rows.jsonl").read_bytes()
        ledger = load_json(campaign.attempt_dir / "embedding_invalid_retry_ledger.json")
        self.assertEqual(ledger["entries"][0]["resolution_status"], "UNRESOLVED_RETRY_PENDING")
        batch_id = caught.exception.batch_id
        invalid_manifest = load_json(
            campaign.attempt_dir
            / f"embedding_batches/{batch_id}/A0/call_manifest.json"
        )
        self.assertEqual(invalid_manifest["analytical_classification"], "INVALID_RUN")

        passing_factory = _Factory(_DeterministicEmbeddingAdapter())
        resumed = self._subset(factory=passing_factory)
        resumed.ensure_embedding_cache()
        retry_manifest = load_json(
            resumed.attempt_dir
            / f"embedding_batches/{batch_id}/A1/call_manifest.json"
        )
        self.assertEqual(retry_manifest["retry_of"], invalid_manifest["call_attempt_id"])
        self.assertEqual(retry_manifest["supersedes_attempt"], invalid_manifest["call_attempt_id"])
        self.assertEqual(retry_manifest["input_sha256"], invalid_manifest["input_sha256"])
        self.assertEqual(retry_manifest["input_texts"], invalid_manifest["input_texts"])
        ledger = load_json(resumed.attempt_dir / "embedding_invalid_retry_ledger.json")
        self.assertEqual(ledger["entries"][0]["retry_call_attempt_id"], retry_manifest["call_attempt_id"])
        self.assertEqual(ledger["entries"][0]["resolution_status"], "RESOLVED_BY_VALID_RETRY")
        self.assertEqual((resumed.attempt_dir / "input_rows.jsonl").read_bytes(), input_before)
        self.assertEqual(len(list(iter_jsonl(resumed.attempt_dir / "input_rows.jsonl"))), 1)

    def test_resume_reuses_hash_verified_completed_embedding_batches(self):
        first_adapter = _QueryDisconnectedAfterDocuments()
        campaign = self._subset(factory=_Factory(first_adapter))
        with self.assertRaises(MemoryCampaignEndpointInvalid) as caught:
            campaign.ensure_embedding_cache()
        self.assertTrue(caught.exception.batch_id.startswith("query-"))
        self.assertEqual([kind for kind, _texts in first_adapter.calls], ["DOCUMENT"])

        resumed_adapter = _DeterministicEmbeddingAdapter()
        resumed = self._subset(factory=_Factory(resumed_adapter))
        resumed.ensure_embedding_cache()
        self.assertEqual([kind for kind, _texts in resumed_adapter.calls], ["QUERY"])
        document_manifests = list(
            (resumed.attempt_dir / "embedding_batches").glob(
                "document-*/A*/call_manifest.json"
            )
        )
        self.assertEqual(len(document_manifests), 1)
        self.assertEqual(load_json(document_manifests[0])["attempt_number"], 0)

    def test_prepared_memory_trace_recovers_result_without_retrieval_recompute(self):
        campaign = self._subset()
        campaign.ensure_embedding_cache()
        production_retrieve = Memory_Agent._retrieve
        calls = []

        def tracked_retrieve(agent, query):
            calls.append(tuple(query))
            return production_retrieve(agent, query)

        from experiments_suite_v2.runners import memory_campaign as campaign_module

        original_append = campaign_module.append_jsonl
        failed_once = {"value": False}

        def interrupted_append(path, record):
            if Path(path).name == "retrieval_trace.jsonl" and not failed_once["value"]:
                failed_once["value"] = True
                raise RuntimeError("simulated crash after durable memory prepare")
            return original_append(path, record)

        with patch.object(Memory_Agent, "_retrieve", new=tracked_retrieve), patch.object(
            campaign_module, "append_jsonl", new=interrupted_append
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                campaign.run_rows(max_new_rows=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(list(iter_jsonl(campaign.attempt_dir / "memory_trace.jsonl"))), 1)
        self.assertFalse((campaign.attempt_dir / "result_rows.jsonl").exists())

        no_live_factory = _Factory(_DisconnectedAdapter())
        resumed = self._subset(factory=no_live_factory)
        with patch.object(Memory_Agent, "_retrieve", new=tracked_retrieve):
            progress = resumed.run_rows(max_new_rows=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(no_live_factory.calls, [])
        self.assertEqual(progress.result_rows, 1)
        self.assertEqual(progress.retrieval_trace_rows, 1)
        summary = self.store._validate_batch_bundle(resumed.attempt_dir)
        self.assertEqual(summary["row_count"], 1)
        prepared = list(iter_jsonl(resumed.attempt_dir / "memory_trace.jsonl"))[0]
        self.assertEqual(
            prepared["production_retrieval_method"],
            "prefmem.agents.memory.Memory_Agent._retrieve",
        )

    def test_cache_corruption_is_detected_before_any_row_resume(self):
        campaign = self._subset()
        campaign.ensure_embedding_cache()
        path = campaign.attempt_dir / "embedding_cache/embedding_cache.npz"
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            campaign.progress()

    def test_selection_is_committed_before_untouched_confirmatory_report(self):
        all_needs = all_memory_grid_needs()
        needs = (all_needs[0], all_needs[20])
        target = (1, 1, 1, 5)
        rows = tuple(
            row
            for row in build_memory_grid_rows(needs)
            if (row.query_count, row.top_k, row.top_cap_k, row.store_size)
            == target
        )
        self.assertEqual([row.split for row in rows], ["DEVELOPMENT", "CONFIRMATORY"])
        campaign = MemoryCampaign(
            store=self.store,
            suite_run_id="memory-run",
            embedding_adapter_factory=_Factory(_DeterministicEmbeddingAdapter()),
            _test_needs=needs,
            _test_rows=rows,
        )
        self.assertTrue(campaign.run_rows().complete)
        selection, confirmation = campaign.write_selection_reports()
        self.assertEqual(selection["source_context_count"], 1)
        self.assertEqual(selection["confirmatory_rows_read"], 0)
        self.assertEqual(
            selection["q1k1_miss_richer_recovery_counterexample"], None
        )
        self.assertFalse(selection["counterexample_observed"])
        self.assertEqual(confirmation["source_context_count"], 1)
        self.assertFalse(confirmation["selection_recomputed_on_confirmatory"])
        self.assertEqual(
            confirmation["selection_commit_sha256"],
            hashlib.sha256(
                (campaign.attempt_dir / "development_selection.json").read_bytes()
            ).hexdigest(),
        )
        # Exercise the immutable RunStore close path on this exact test window;
        # production construction itself still enforces 19,200 rows.
        from experiments_suite_v2.runners import memory_campaign as campaign_module

        campaign.full_registry = True
        with patch.multiple(
            campaign_module,
            EXPECTED_RESULT_ROWS=2,
            EXPECTED_PARAMETER_CELLS=1,
            EXPECTED_DEVELOPMENT_NEEDS=1,
            EXPECTED_CONFIRMATORY_NEEDS=1,
        ):
            artifact_manifest = campaign.finalize()
        self.assertEqual(artifact_manifest["schema_version"], "prefmem.artifact-manifest.v2")
        self.assertEqual(
            load_json(campaign.attempt_dir / "attempt_manifest.json")["status"],
            "FINALIZED",
        )

    def test_development_selector_is_deterministic_and_rejects_confirmatory_leakage(self):
        baseline = (1, 1, 1, 5)
        richer = (3, 3, 5, 5)
        development = []
        for context_id in ("MGQ010", "MGQ002"):
            development.extend(
                (
                    _scored_row(
                        context_id=context_id,
                        configuration=baseline,
                        recall=0.0,
                        mrr=0.0,
                        precision=0.0,
                        returned=("distractor",),
                    ),
                    _scored_row(
                        context_id=context_id,
                        configuration=richer,
                        recall=1.0,
                        mrr=1.0,
                        precision=1.0,
                        returned=("required",),
                    ),
                )
            )
        first = select_development_configuration(
            development, require_complete_grid=False
        )
        second = select_development_configuration(
            tuple(reversed(development)), require_complete_grid=False
        )
        self.assertEqual(first["selected_configuration"], second["selected_configuration"])
        self.assertEqual(first["selected_configuration"], {
            "query_count": 3,
            "top_k": 3,
            "top_cap_k": 5,
            "store_size": 5,
        })
        self.assertEqual(first["selection_rule_sha256"], DEVELOPMENT_SELECTION_RULE_SHA256)
        self.assertEqual(first["confirmatory_rows_read"], 0)
        counterexample = first_q1k1_recovery_counterexample(
            development, first["selected_configuration"]
        )
        self.assertEqual(counterexample["context_id"], "MGQ002")
        self.assertEqual(counterexample["baseline_returned_ids"], ["distractor"])
        self.assertEqual(counterexample["selected_returned_ids"], ["required"])

        contaminated = development + [
            _scored_row(
                context_id="MGQ041",
                configuration=richer,
                recall=0.0,
                mrr=0.0,
                precision=0.0,
                split="CONFIRMATORY",
            )
        ]
        with self.assertRaisesRegex(MemoryCampaignError, "DEVELOPMENT rows only"):
            select_development_configuration(
                contaminated, require_complete_grid=False
            )

    def test_selection_rule_accepts_exact_40_by_240_development_lattice(self):
        rows = []
        for q, k, cap, size in itertools.product(
            (1, 3, 5, 10), (1, 3, 5, 10), (1, 3, 5, 10, 20), (5, 25, 100)
        ):
            for context_number in range(1, EXPECTED_DEVELOPMENT_NEEDS + 1):
                rows.append(
                    _scored_row(
                        context_id=f"MGQ{context_number:03d}",
                        configuration=(q, k, cap, size),
                        recall=1.0,
                        mrr=1.0,
                        precision=1.0,
                    )
                )
        selection = select_development_configuration(rows)
        self.assertEqual(len(rows), 9_600)
        self.assertEqual(len(selection["ranked_cells"]), EXPECTED_PARAMETER_CELLS)
        self.assertEqual(selection["source_context_count"], EXPECTED_DEVELOPMENT_NEEDS)
        self.assertEqual(
            selection["selected_configuration"],
            {"query_count": 1, "top_k": 1, "top_cap_k": 1, "store_size": 5},
        )

    def test_frozen_split_dimensions_remain_40_and_40(self):
        needs = all_memory_grid_needs()
        self.assertEqual(sum(need.split == "DEVELOPMENT" for need in needs), EXPECTED_DEVELOPMENT_NEEDS)
        self.assertEqual(sum(need.split == "CONFIRMATORY" for need in needs), EXPECTED_CONFIRMATORY_NEEDS)


if __name__ == "__main__":
    unittest.main()
