from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from prefmem.agents.contracts import (
    EpisodeRecord,
    MemoryContext,
    MemoryRetrievalRequest,
    MemoryStatus,
    PreferenceRecord,
    PreferenceStatus,
)
from prefmem.memory.embeddings import EmbeddingServiceError
from prefmem.memory.service import (
    MemoryAuthorizationError,
    PrefMemMemoryService,
)
from prefmem.memory.store import MarkdownMemoryStore, MemoryFormatError


class FakeEmbeddings:
    def __init__(self) -> None:
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        self.query = text
        return [1.0, 0.0]

    def embed_documents(self, documents) -> list[list[float]]:
        self.documents = list(documents)
        return [[1.0, 0.0] for _ in self.documents]


class UnavailableEmbeddings(FakeEmbeddings):
    def embed_documents(self, documents) -> list[list[float]]:
        raise EmbeddingServiceError("embedding sidecar is offline")


class FakeMemoryReasoner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, **kwargs) -> MemoryContext:
        self.calls.append(kwargs)
        candidates = kwargs["preference_candidates"]
        return MemoryContext(
            status=MemoryStatus.AVAILABLE,
            request_id=kwargs["request"].request_id,
            relevant_preferences=[
                {
                    "record_id": candidates[0]["record_id"],
                    "relation": "MATCH",
                }
            ],
        )


class MarkdownMemoryStoreTests(unittest.TestCase):
    def test_default_path_rejects_dot_segments_and_avoids_sanitized_collisions(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            MarkdownMemoryStore.default_path("..")
        self.assertNotEqual(
            MarkdownMemoryStore.default_path("alice/bob"),
            MarkdownMemoryStore.default_path("alice-bob"),
        )

    def test_preference_and_episode_round_trip_as_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MarkdownMemoryStore(Path(temporary) / "preferences.md")
            preference = PreferenceRecord(
                id="pref-1",
                username="alice",
                statement="Use the blue mug by default.",
            )
            episode = EpisodeRecord(
                id="episode-1",
                username="alice",
                request="Place the banana in the mug.",
                summary="A plan was produced but not executed.",
                result={"status": "PLANNED_NOT_EXECUTED"},
            )

            store.save_preference(preference)
            store.save_episode(episode)

            self.assertEqual(store.load_preferences(), [preference])
            self.assertEqual(store.load_episodes(), [episode])
            preference_text = store.preference_path.read_text(encoding="utf-8")
            history_text = store.history_path.read_text(encoding="utf-8")
            self.assertIn("```prefmem-preference", preference_text)
            self.assertIn("Use the blue mug", preference_text)
            self.assertIn("```prefmem-episode", history_text)

    def test_distinct_store_instances_preserve_each_others_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preferences.md"
            first = MarkdownMemoryStore(path)
            second = MarkdownMemoryStore(path)

            first.save_preference(
                PreferenceRecord(
                    id="pref-1",
                    username="alice",
                    statement="Use the blue mug.",
                )
            )
            second.save_preference(
                PreferenceRecord(
                    id="pref-2",
                    username="alice",
                    statement="Place books on the left.",
                )
            )

            self.assertEqual(
                {record.id for record in first.load_preferences()},
                {"pref-1", "pref-2"},
            )

    def test_episode_ids_are_immutable_but_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MarkdownMemoryStore(Path(temporary) / "preferences.md")
            episode = EpisodeRecord(
                id="episode-1",
                username="alice",
                request="Place the banana.",
                result={"status": "SUCCESS"},
            )
            store.save_episode(episode)
            store.save_episode(episode)

            with self.assertRaisesRegex(ValueError, "immutable"):
                store.save_episode(
                    episode.model_copy(
                        update={"result": {"status": "FAILURE"}}
                    )
                )

    def test_invalid_canonical_record_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preferences.md"
            path.write_text(
                "# Memory\n\n```prefmem-preference\n{\"id\": 3}\n```\n",
                encoding="utf-8",
            )
            store = MarkdownMemoryStore(path)

            with self.assertRaises(MemoryFormatError):
                store.load_preferences()

    def test_truncated_canonical_record_is_rejected_before_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preferences.md"
            original = (
                "# Memory\n\n```prefmem-preference\n"
                '{"id": "unfinished"}\n'
            )
            path.write_text(original, encoding="utf-8")
            store = MarkdownMemoryStore(path)

            with self.assertRaises(MemoryFormatError):
                store.save_preference(
                    PreferenceRecord(
                        id="pref-new",
                        username="alice",
                        statement="Use the blue mug.",
                    )
                )

            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_malformed_and_wrong_kind_fences_are_rejected(self) -> None:
        invalid_documents = (
            "```prefmem-preference\n{not-json}\n```\n",
            "```prefmem-preference extra\n{}\n```\n",
            "```prefmem-episode\n{}\n```\n",
        )
        for text in invalid_documents:
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "preferences.md"
                    path.write_text(text, encoding="utf-8")

                    with self.assertRaises(MemoryFormatError):
                        MarkdownMemoryStore(path).load_preferences()

    def test_duplicate_record_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preferences.md"
            record = PreferenceRecord(
                id="pref-duplicate",
                username="alice",
                statement="Use the blue mug.",
            )
            block = (
                "```prefmem-preference\n"
                f"{record.model_dump_json()}\n"
                "```"
            )
            path.write_text(f"{block}\n\n{block}\n", encoding="utf-8")

            with self.assertRaises(MemoryFormatError):
                MarkdownMemoryStore(path).load_preferences()

    def test_prefmem_marker_inside_json_string_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MarkdownMemoryStore(Path(temporary) / "preferences.md")
            record = PreferenceRecord(
                id="pref-marker",
                username="alice",
                statement="Treat ```prefmem-preference as text.",
            )

            store.save_preference(record)

            self.assertEqual(store.load_preferences(), [record])


class PrefMemMemoryServiceTests(unittest.TestCase):
    def make_service(
        self,
        root: Path,
        *,
        reasoner=None,
        embeddings=None,
        semantic_duplicate_threshold: float | None = 0.995,
    ) -> PrefMemMemoryService:
        return PrefMemMemoryService(
            MarkdownMemoryStore(root / "preferences.md"),
            embeddings or FakeEmbeddings(),
            semantic_reasoner=reasoner,
            minimum_similarity=0.0,
            semantic_duplicate_threshold=semantic_duplicate_threshold,
        )

    def test_write_requires_host_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))

            with self.assertRaises(MemoryAuthorizationError):
                service.remember_preference(
                    username="alice",
                    statement="Use the blue mug.",
                    authorized=False,
                )

    def test_empty_store_does_not_require_embedding_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))

            context = service.retrieve(
                username="alice",
                request=MemoryRetrievalRequest(
                    search_text="usual mug",
                    reason_code="PREFERENCE_SENSITIVE",
                ),
            )

            self.assertEqual(context.status, MemoryStatus.EMPTY)
            self.assertEqual(service.embeddings.query_calls, 0)

    def test_exact_duplicate_preference_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            first = service.remember_preference(
                username="alice",
                statement="Use the blue mug by default.",
                authorized=True,
            )

            duplicate = service.remember_preference(
                username="alice",
                statement="  use the BLUE mug by default. ",
                authorized=True,
            )

            self.assertEqual(duplicate.id, first.id)
            self.assertEqual(len(service.store.load_preferences()), 1)

    def test_conservative_semantic_duplicate_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            first = service.remember_preference(
                username="alice",
                statement="Use the blue mug by default.",
                authorized=True,
            )

            duplicate = service.remember_preference(
                username="alice",
                statement="Use the blue mug by default!",
                authorized=True,
            )

            self.assertEqual(duplicate.id, first.id)
            self.assertEqual(len(service.store.load_preferences()), 1)

    def test_high_cosine_role_swap_is_not_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.remember_preference(
                username="alice",
                statement=(
                    "Put books on the left and electronics on the right."
                ),
                authorized=True,
            )

            second = service.remember_preference(
                username="alice",
                statement=(
                    "Put electronics on the left and books on the right."
                ),
                authorized=True,
            )

            self.assertIsNotNone(second)
            self.assertEqual(len(service.store.load_preferences()), 2)

    def test_embedding_outage_does_not_discard_authorized_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(
                Path(temporary),
                embeddings=UnavailableEmbeddings(),
            )
            service.remember_preference(
                username="alice",
                statement="Use the blue mug by default.",
                authorized=True,
            )

            service.remember_preference(
                username="alice",
                statement="Use the blue mug by default!",
                authorized=True,
            )

            self.assertEqual(len(service.store.load_preferences()), 2)

    def test_semantic_duplicate_check_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(
                Path(temporary),
                semantic_duplicate_threshold=None,
            )
            service.remember_preference(
                username="alice",
                statement="Use the blue mug by default.",
                authorized=True,
            )
            service.remember_preference(
                username="alice",
                statement="Use the blue mug by default!",
                authorized=True,
            )

            self.assertEqual(len(service.store.load_preferences()), 2)

    def test_retrieval_uses_embeddings_then_semantic_reasoner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reasoner = FakeMemoryReasoner()
            service = self.make_service(Path(temporary), reasoner=reasoner)
            service.remember_preference(
                username="alice",
                statement="Use the blue mug by default.",
                authorized=True,
            )

            context = service.retrieve(
                username="alice",
                request=MemoryRetrievalRequest(
                    request_id="request-1",
                    search_text="Which mug should I use?",
                    reason_code="PREFERENCE_SENSITIVE",
                ),
            )

            self.assertEqual(context.status, MemoryStatus.AVAILABLE)
            self.assertEqual(len(reasoner.calls), 1)
            candidate = reasoner.calls[0]["preference_candidates"][0]
            self.assertTrue(candidate["record_id"].startswith("pref-"))
            self.assertIn("embedding_similarity", candidate)

    def test_update_is_one_atomic_revision_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            original = service.remember_preference(
                username="alice",
                statement="Use the blue mug.",
                authorized=True,
            )

            replacement = service.update_preference(
                username="alice",
                preference_id=original.id,
                statement="Use the white mug.",
                authorized=True,
            )

            all_records = service.store.load_preferences(active_only=False)
            by_id = {record.id: record for record in all_records}
            self.assertEqual(
                by_id[original.id].status,
                PreferenceStatus.REVOKED,
            )
            self.assertEqual(replacement.supersedes_id, original.id)
            self.assertEqual(
                service.store.load_preferences(),
                [replacement],
            )


if __name__ == "__main__":
    unittest.main()
