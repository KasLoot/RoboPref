from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory.models import ConsentEvidence
from memory.repositories import (
    HistoryRepository,
    MemoryRepositoryError,
    PreferenceRepository,
)


def future_consent(turn: str = "turn-1") -> ConsentEvidence:
    return ConsentEvidence(
        kind="explicit_future_language",
        quote="Remember this from now on.",
        turn_id=turn,
        episode_id="episode-1",
        authorized_action="UPSERT",
        proposal={"instruction": "Remember this from now on."},
    )


def preference(statement: str, *, scope: str = "block stacks") -> dict[str, object]:
    return {
        "statement": statement,
        "scope": scope,
        "applicability": {"objects": "red, green, and blue blocks or cubes"},
        "structured_value": {"bottom": "red", "middle": "green", "top": "blue"},
    }


def verified_merge(*source_ids: str) -> dict[str, object]:
    return {
        "source_preference_ids": list(source_ids),
        "equivalent": True,
        "scope_preserved": True,
        "applicability_preserved": True,
        "confidence": 0.98,
        "reason": "Independent semantic verification confirmed equivalence.",
    }


class HistoryRepositoryTests(unittest.TestCase):
    def test_append_is_idempotent_and_changed_replay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = HistoryRepository(Path(directory) / "history.json")
            episode = {
                "episode_id": "episode-1",
                "user_id": "participant-a",
                "summary": "Stacked the blocks in RGB order.",
            }

            first = repository.append(episode)
            second = repository.append(episode)

            self.assertEqual(first, second)
            self.assertEqual(repository.version, 1)
            self.assertEqual(len(repository.list_episodes("participant-a")), 1)

            changed = {**episode, "summary": "A changed retry."}
            with self.assertRaisesRegex(MemoryRepositoryError, "idempotency conflict"):
                repository.append(changed)
            self.assertEqual(repository.version, 1)
            self.assertEqual(repository.list_episodes("participant-a")[0]["summary"], episode["summary"])


class PreferenceRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = PreferenceRepository(
            Path(self.temporary_directory.name) / "preferences.json"
        )

    def _add(self, identifier: str, statement: str, turn: str) -> str:
        transaction = self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id=f"txn-{identifier}",
            operations=[
                {
                    "operation_id": f"op-{identifier}",
                    "action": "ADD",
                    "preference": preference(statement),
                    "evidence": [{"episode_id": f"episode-{turn}"}],
                }
            ],
            authorization=future_consent(turn),
        )
        return transaction["results"][0]["preference_id"]

    def test_add_requires_user_memory_authorization(self) -> None:
        maintenance = ConsentEvidence(
            kind="memory_maintenance",
            quote="Internal maintenance is not user consent.",
            turn_id="maintenance-1",
            authorized_action="COMPACT",
        )
        with self.assertRaises(MemoryRepositoryError):
            self.repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-unconsented",
                operations=[
                    {
                        "operation_id": "op-unconsented",
                        "action": "ADD",
                        "preference": preference("Use RGB for block towers."),
                    }
                ],
                authorization=maintenance,
            )
        self.assertEqual(self.repository.list_preferences("participant-a"), [])
        self.assertEqual(self.repository.version, 0)

    def test_delete_requires_explicit_forget_confirmation(self) -> None:
        preference_id = self._add("add", "Use RGB for block towers.", "turn-add")
        ordinary_confirmation = ConsentEvidence(
            kind="memory_confirmation",
            quote="Yes.",
            turn_id="turn-delete-wrong",
            episode_id="episode-2",
            prompt_id="prompt-memory",
            authorized_action="UPSERT",
            displayed_question="Should I remember this preference?",
            proposal={"instruction": "Remember the preference."},
        )
        with self.assertRaises(MemoryRepositoryError):
            self.repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-delete-wrong",
                operations=[
                    {
                        "operation_id": "op-delete-wrong",
                        "action": "DELETE",
                        "target_preference_id": preference_id,
                    }
                ],
                authorization=ordinary_confirmation,
            )
        self.assertEqual(len(self.repository.list_preferences("participant-a")), 1)

        forget = ConsentEvidence(
            kind="forget_confirmation",
            quote="Forget that saved order.",
            turn_id="turn-delete",
            episode_id="episode-2",
            authorized_action="DELETE",
            proposal={"instruction": "Forget the saved order."},
        )
        self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id="txn-delete",
            operations=[
                {
                    "operation_id": "op-delete",
                    "action": "DELETE",
                    "target_preference_id": preference_id,
                }
            ],
            authorization=forget,
        )
        self.assertEqual(self.repository.list_preferences("participant-a"), [])
        tombstone = self.repository.get(preference_id, "participant-a")
        self.assertEqual(tombstone["status"], "deleted")
        self.assertEqual(tombstone["consent"][-1]["kind"], "forget_confirmation")

    def test_invalid_multi_operation_transaction_is_atomic(self) -> None:
        with self.assertRaisesRegex(MemoryRepositoryError, "Unknown preference"):
            self.repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-atomic",
                operations=[
                    {
                        "operation_id": "op-valid-add",
                        "action": "ADD",
                        "preference": preference("Use RGB for block towers."),
                    },
                    {
                        "operation_id": "op-invalid-update",
                        "action": "UPDATE",
                        "target_preference_id": "pref-does-not-exist",
                        "preference": preference("Use BGR for block towers."),
                    },
                ],
                authorization=future_consent(),
            )

        self.assertEqual(self.repository.list_preferences("participant-a", include_inactive=True), [])
        self.assertEqual(self.repository.version, 0)

    def test_transaction_replay_is_idempotent_but_changed_replay_fails(self) -> None:
        operation = {
            "operation_id": "op-add",
            "action": "ADD",
            "preference": preference("Use RGB for block towers."),
        }
        first = self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id="txn-replay",
            operations=[operation],
            authorization=future_consent(),
        )
        second = self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id="txn-replay",
            operations=[operation],
            authorization=future_consent(),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(self.repository.list_preferences("participant-a")), 1)
        self.assertEqual(self.repository.version, 1)

        changed = {**operation, "preference": preference("Use BGR for block towers.")}
        with self.assertRaisesRegex(MemoryRepositoryError, "idempotency conflict"):
            self.repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-replay",
                operations=[changed],
                authorization=future_consent(),
            )

    def test_lossless_semantic_merge_preserves_sources_and_is_reversible(self) -> None:
        rgb_id = self._add(
            "rgb",
            "Stack red, green, and blue cubes from bottom to top.",
            "turn-rgb",
        )
        positions_id = self._add(
            "positions",
            "For blocks, put blue on top, green in the middle, and red on the bottom.",
            "turn-positions",
        )
        source_before = {
            item["id"]: item
            for item in self.repository.list_preferences(
                "participant-a", include_inactive=True
            )
        }

        transaction = self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id="txn-merge",
            operations=[
                {
                    "operation_id": "op-merge",
                    "action": "MERGE",
                    "source_preference_ids": [rgb_id, positions_id],
                    "lossless": True,
                    "confidence": 0.98,
                    "reason": "The two statements describe the same physical order.",
                    "semantic_verification": verified_merge(rgb_id, positions_id),
                    "preference": preference(
                        "When stacking these blocks or cubes, put red at the bottom, "
                        "green in the middle, and blue on top."
                    ),
                }
            ],
            authorization=ConsentEvidence(
                kind="memory_maintenance",
                quote="Reversible lossless semantic compaction.",
                turn_id="maintenance-merge",
                authorized_action="COMPACT",
            ),
        )

        canonical_id = transaction["results"][0]["preference_id"]
        all_records = {
            item["id"]: item
            for item in self.repository.list_preferences(
                "participant-a", include_inactive=True
            )
        }
        self.assertEqual(set(all_records), {rgb_id, positions_id, canonical_id})
        self.assertEqual(self.repository.list_preferences("participant-a")[0]["id"], canonical_id)
        for source_id in (rgb_id, positions_id):
            self.assertEqual(all_records[source_id]["status"], "merged")
            self.assertEqual(all_records[source_id]["merged_into"], canonical_id)
            self.assertEqual(
                all_records[source_id]["statement"], source_before[source_id]["statement"]
            )
        canonical = all_records[canonical_id]
        self.assertCountEqual(
            canonical["lineage"]["merged_from"], [rgb_id, positions_id]
        )
        self.assertEqual(len(canonical["consent"]), 2)
        self.assertEqual(len(canonical["evidence"]), 2)

    def test_merge_into_existing_target_keeps_its_premerge_revision(self) -> None:
        """Choosing a source as canonical must not destroy that source's old wording."""
        cube_id = self._add(
            "cube-target",
            "Stack the coloured cubes RGB from bottom to top.",
            "turn-cube-target",
        )
        block_id = self._add(
            "block-source",
            "Put blue on top, green in the middle, and red under the blocks.",
            "turn-block-source",
        )
        target_before = self.repository.get(cube_id, "participant-a")

        self.repository.apply_transaction(
            user_id="participant-a",
            transaction_id="txn-merge-existing-target",
            operations=[
                {
                    "operation_id": "op-merge-existing-target",
                    "action": "MERGE",
                    "source_preference_ids": [cube_id, block_id],
                    "target_preference_id": cube_id,
                    "lossless": True,
                    "confidence": 0.97,
                    "semantic_verification": verified_merge(cube_id, block_id),
                    "preference": preference(
                        "For these blocks or cubes, place red at the bottom, green in "
                        "the middle, and blue on top."
                    ),
                }
            ],
            authorization=ConsentEvidence(
                kind="memory_maintenance",
                quote="Reversible lossless semantic compaction.",
                turn_id="maintenance-existing-target",
                authorized_action="COMPACT",
            ),
        )

        canonical = self.repository.get(cube_id, "participant-a")
        prior_versions = canonical.get("versions", [])
        self.assertTrue(
            any(
                version.get("statement") == target_before["statement"]
                for version in prior_versions
            ),
            "An in-place semantic merge must retain the canonical source's pre-merge revision.",
        )
        redirected = self.repository.get(block_id, "participant-a")
        self.assertEqual(redirected["status"], "merged")
        self.assertEqual(redirected["merged_into"], cube_id)


if __name__ == "__main__":
    unittest.main()
