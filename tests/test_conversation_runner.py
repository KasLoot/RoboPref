from __future__ import annotations

import unittest
from pathlib import Path

from dataset.benchmark import BenchmarkEpisode
from simulation.benchmark.conversation_cases import load_episode_catalog
from simulation.benchmark.conversation_models import CommandSpec
from simulation.benchmark.conversation_runner import (
    ConversationRunError,
    _ensure_isolated,
    _memory_delta,
    scripted_reply,
)
from simulation.benchmark.executor import CounterfactualSequenceExecutor


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "sim_datasets"


class ScriptedUserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_episode_catalog(ROOT)
        cls.descriptor = next(
            item
            for item in cls.catalog
            if item.family == "category_sort"
            and item.target_id == "printed_left"
            and item.outcome == "success"
            and item.control_kind is None
        )

    def command(self) -> CommandSpec:
        return CommandSpec(
            command_id="resolve-category",
            episode_id=self.descriptor.scenario_id,
            query="Tidy up the table.",
            target_id="printed_left",
            query_kind="ambiguous",
            response_policy="target",
            expected_initial_modes=("ASK", "CONFIRM"),
        )

    @staticmethod
    def result(printed_side: str, prose: str = "Should I do this?") -> dict:
        electronic_side = "right" if printed_side == "left" else "left"
        return {
            "hri": {
                "mode": "CONFIRM",
                "user_message": prose,
                "pending_question": {
                    "kind": "TASK_CONFIRMATION",
                    "payload": {
                        "proposed_task": {
                            "confirmed_intent": prose,
                            "task_type": "sort_categories",
                            "objects": ["printed items", "electronic devices"],
                            "parameters": {
                                "category_to_side": {
                                    "printed": printed_side,
                                    "electronic": electronic_side,
                                }
                            },
                        }
                    },
                },
            },
            "awaiting_user": True,
        }

    def test_yes_is_sent_only_for_the_structured_desired_proposal(self) -> None:
        reply, evidence = scripted_reply(
            self.command(), self.descriptor.manifest, self.result("left")
        )
        self.assertEqual(reply, "Yes.")
        self.assertEqual(evidence["status"], "ACCEPTED_DESIRED")

    def test_registered_opposite_is_explicitly_rejected(self) -> None:
        reply, evidence = scripted_reply(
            self.command(), self.descriptor.manifest, self.result("right")
        )
        self.assertEqual(reply, "No, do the opposite.")
        self.assertEqual(evidence["status"], "REJECTED_OPPOSITE")

    def test_assistant_prose_does_not_change_structured_reply(self) -> None:
        first, _ = scripted_reply(
            self.command(),
            self.descriptor.manifest,
            self.result("left", "Printed left?"),
        )
        second, _ = scripted_reply(
            self.command(),
            self.descriptor.manifest,
            self.result("left", "Completely unrelated wording."),
        )
        self.assertEqual(first, second)

    def test_missing_proposal_never_receives_blind_affirmation(self) -> None:
        reply, evidence = scripted_reply(
            self.command(),
            self.descriptor.manifest,
            {
                "hri": {
                    "mode": "CONFIRM",
                    "pending_question": {
                        "kind": "TASK_CONFIRMATION",
                        "payload": {},
                    },
                },
                "awaiting_user": True,
            },
        )
        self.assertTrue(reply.startswith("No. Put printed items"))
        self.assertEqual(evidence["status"], "SAFE_EXPLICIT_CORRECTION")


class CounterfactualSequenceExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog = load_episode_catalog(ROOT)
        cls.failure = next(item for item in catalog if item.outcome == "partial")
        cls.success = next(
            item
            for item in catalog
            if item.counterfactual_group_id == cls.failure.counterfactual_group_id
            and item.outcome == "success"
            and item.control_kind is None
        )

    def test_recovery_endpoint_is_revealed_only_after_recovery_action(self) -> None:
        failure = BenchmarkEpisode.from_path(self.failure.path, ROOT)
        success = BenchmarkEpisode.from_path(self.success.path, ROOT)
        executor = CounterfactualSequenceExecutor(failure)
        executor.configure_recovery(success)

        self.assertEqual(executor.observe(), str(failure.episode.final_frame))
        executor.execute_subtask({}, index=1, attempt=2)
        self.assertEqual(executor.observe(), str(success.episode.final_frame))

        executor.configure_recovery(success)
        self.assertEqual(executor.observe(), str(failure.episode.final_frame))
        self.assertEqual(
            executor.reobserve(index=1), str(success.episode.final_frame)
        )


class MemoryAuditRegressionTests(unittest.TestCase):
    def test_same_count_content_mutation_is_detected_by_state_digest(self) -> None:
        before = {
            "available": True,
            "history_count": 1,
            "preference_count": 1,
            "outbox_count": 0,
            "history_state_digest": "history-before",
            "preference_state_digest": "preference-before",
            "outbox_state_digest": "outbox-stable",
            "history_version": 2,
            "preference_version": 4,
        }
        after = {
            **before,
            "history_state_digest": "history-after",
            "preference_state_digest": "preference-after",
            "history_version": 3,
            "preference_version": 5,
        }

        delta = _memory_delta(before, after)

        self.assertEqual(delta["history_delta"], 0)
        self.assertEqual(delta["preference_delta"], 0)
        self.assertTrue(delta["history_state_changed"])
        self.assertTrue(delta["preference_state_changed"])

    def test_zero_records_with_stale_summary_is_not_cold_memory(self) -> None:
        snapshot = {
            "available": True,
            "history_count": 0,
            "preference_count": 0,
            "outbox_count": 0,
            "history_summary": {
                "text": "The participant previously preferred RGB.",
                "source_episode_ids": [],
            },
        }

        with self.assertRaises(ConversationRunError):
            _ensure_isolated(snapshot, user_id="participant-a")


if __name__ == "__main__":
    unittest.main()
