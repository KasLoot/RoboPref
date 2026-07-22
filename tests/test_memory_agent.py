from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agents.configs import AgentModelConfig
from agents.memory import FORBIDDEN_HISTORY_KEYS, MemoryAgent
from memory.models import ConsentEvidence, MemoryQuery
from memory.repositories import HistoryRepository, PreferenceRepository
from tests.fakes import ScriptedJsonModel


ROOT = Path(__file__).resolve().parents[1]


def memory_config() -> AgentModelConfig:
    return AgentModelConfig(
        model="offline-only",
        system_prompt_path=str(ROOT / "prompt" / "memory" / "prompt-v2.md"),
    )


def consent(turn: str) -> ConsentEvidence:
    return ConsentEvidence(
        kind="explicit_future_language",
        quote="Remember this order for future stacks.",
        turn_id=turn,
        episode_id=f"episode-{turn}",
        authorized_action="UPSERT",
        proposal={"instruction": "Remember this order for future stacks."},
    )


def add_preference(
    repository: PreferenceRepository,
    *,
    transaction_id: str,
    statement: str,
    evidence: str,
) -> str:
    transaction = repository.apply_transaction(
        user_id="participant-a",
        transaction_id=transaction_id,
        operations=[
            {
                "operation_id": f"{transaction_id}-op",
                "action": "ADD",
                "preference": {
                    "statement": statement,
                    "scope": "contextual coloured-object stacking",
                    "applicability": {"scene": "three red, green, and blue rigid objects"},
                    "structured_value": {
                        "bottom": "red",
                        "middle": "green",
                        "top": "blue",
                    },
                },
                "evidence": [{"episode_id": evidence}],
            }
        ],
        authorization=consent(transaction_id),
    )
    return transaction["results"][0]["preference_id"]


class MemoryAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        directory = Path(self.temporary_directory.name)
        self.history = HistoryRepository(directory / "history.json")
        self.preferences = PreferenceRepository(directory / "preferences.json")

    def test_semantic_retrieval_has_no_block_cube_or_stack_stacking_lexical_gate(self) -> None:
        preference_id = add_preference(
            self.preferences,
            transaction_id="txn-cubes",
            statement=(
                "When building a tower from the coloured cubes, place red lowest, "
                "green centrally, and blue highest."
            ),
            evidence="episode-cubes",
        )

        def retrieve(payload: dict[str, Any]) -> dict[str, Any]:
            self.assertEqual(payload["query"]["request"], "Stack the blocks")
            self.assertEqual(payload["preferences"][0]["id"], preference_id)
            return {
                "matches": [
                    {
                        "preference_id": preference_id,
                        "relation": "MATCH",
                        "confidence": 0.96,
                        "reason": "Blocks and cubes denote the same visible objects here.",
                        "applicable_value": {
                            "bottom": "red",
                            "middle": "green",
                            "top": "blue",
                        },
                    }
                ]
            }

        model = ScriptedJsonModel({"retrieve_preferences": retrieve})
        agent = MemoryAgent(
            memory_config(), self.history, self.preferences, model=model
        )

        matches = agent.get_relevant_preference_memory(
            MemoryQuery(
                user_id="participant-a",
                request="Stack the blocks",
                scene={"objects": ["red block", "green block", "blue block"]},
            )
        )

        self.assertEqual([item["id"] for item in matches], [preference_id])
        self.assertEqual(matches[0]["match"]["relation"], "MATCH")
        self.assertEqual(matches[0]["match"]["applicable_value"]["top"], "blue")
        self.assertEqual(len(model.calls_for("retrieve_preferences")), 1)

    def test_scripted_vlm_compaction_merges_cube_and_block_phrasings(self) -> None:
        cube_id = add_preference(
            self.preferences,
            transaction_id="txn-cube",
            statement="Stack the cubes RGB from bottom to top.",
            evidence="episode-cube",
        )
        block_id = add_preference(
            self.preferences,
            transaction_id="txn-block",
            statement="For the blocks, blue is top, green middle, and red bottom.",
            evidence="episode-block",
        )

        def compact(payload: dict[str, Any]) -> dict[str, Any]:
            self.assertCountEqual(
                [item["id"] for item in payload["preferences"]],
                [cube_id, block_id],
            )
            return {
                "operations": [
                    {
                        "action": "MERGE",
                        "source_preference_ids": [cube_id, block_id],
                        "lossless": True,
                        "confidence": 0.97,
                        "reason": "Synonymous objects and equivalent inverse descriptions.",
                        "preference": {
                            "statement": (
                                "When stacking the red, green, and blue blocks or cubes, "
                                "put red at the bottom, green in the middle, and blue on top."
                            ),
                            "scope": "contextual coloured-object stacking",
                            "applicability": {
                                "scene": "three red, green, and blue rigid objects"
                            },
                            "structured_value": {
                                "bottom": "red",
                                "middle": "green",
                                "top": "blue",
                            },
                        },
                    }
                ]
            }

        model = ScriptedJsonModel(
            {
                "compact_preferences": compact,
                "verify_preference_merge": lambda payload: {
                    "source_preference_ids": [
                        item["id"] for item in payload["source_preferences"]
                    ],
                    "equivalent": True,
                    "scope_preserved": True,
                    "applicability_preserved": True,
                    "confidence": 0.96,
                    "reason": "The result preserves both meanings and scopes.",
                },
            }
        )
        agent = MemoryAgent(
            memory_config(), self.history, self.preferences, model=model
        )

        result = agent.compact_preference_memory("participant-a")

        self.assertEqual(result["status"], "COMPACTED")
        active = self.preferences.list_preferences("participant-a")
        self.assertEqual(len(active), 1)
        self.assertIn("blocks or cubes", active[0]["statement"])
        inactive = {
            item["id"]: item
            for item in self.preferences.list_preferences(
                "participant-a", include_inactive=True
            )
            if item["id"] in {cube_id, block_id}
        }
        self.assertEqual(inactive[cube_id]["status"], "merged")
        self.assertEqual(inactive[block_id]["status"], "merged")

    def test_history_update_strips_reasoning_even_if_summarizer_reintroduces_it(self) -> None:
        def summarize(payload: dict[str, Any]) -> dict[str, Any]:
            conversation = payload["conversation"]
            self.assertNotIn("trace", conversation)
            self.assertNotIn("planning_status", conversation)
            self.assertNotIn("reasoning", conversation["execution"][0])
            return {
                "episode": {
                    "summary": "The agents attempted the requested RGB stack.",
                    "user_request": conversation["user_request"],
                    "resolved_task": conversation["resolved_task"],
                    "actions": conversation["actions"],
                    "execution": [
                        {
                            "status": "COMPLETED",
                            "trace": "model tried to reintroduce private reasoning",
                        }
                    ],
                    "validation": conversation["validation"],
                    "user_choices": conversation["user_choices"],
                    "user_visible_result": conversation["user_visible_result"],
                    "trace": "also forbidden at the episode root",
                    "planning_status": "READY",
                    "task_complete": True,
                }
            }

        def compact_history(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "summary": "One RGB block-stacking attempt was recorded.",
                "source_episode_ids": [payload["episodes"][0]["episode_id"]],
            }

        model = ScriptedJsonModel(
            {
                "summarize_history_episode": summarize,
                "compact_history": compact_history,
            }
        )
        agent = MemoryAgent(
            memory_config(), self.history, self.preferences, model=model
        )
        source = {
            "episode_id": "episode-history-1",
            "user_id": "participant-a",
            "session_id": "session-1",
            "user_request": "Stack the blocks.",
            "resolved_task": {"confirmed_intent": "Stack RGB bottom-to-top."},
            "actions": [{"task_instruction": "Place green on red."}],
            "execution": [
                {
                    "status": "COMPLETED",
                    "reasoning": "hidden execution reasoning",
                }
            ],
            "validation": [
                {
                    "outcome": "UNKNOWN",
                    "user_message": "The validator reported uncertain contact.",
                    "validator_confidence": 0.4,
                }
            ],
            "user_choices": {"stack_order": ["red", "green", "blue"]},
            "user_visible_result": "The final state could not be verified.",
            "trace": ["hidden HRI trace"],
            "planning_status": "READY",
            "task_complete": False,
            "scene_inventory": ["large", "redundant", "inventory"],
        }

        result = agent.update_history_memory(source)
        stored = result["episode"]

        def assert_sanitized(value: Any) -> None:
            if isinstance(value, dict):
                self.assertTrue(
                    FORBIDDEN_HISTORY_KEYS.isdisjoint(key.lower() for key in value),
                    value,
                )
                for child in value.values():
                    assert_sanitized(child)
            elif isinstance(value, list):
                for child in value:
                    assert_sanitized(child)

        assert_sanitized(stored)
        self.assertEqual(stored["validation"][0]["outcome"], "UNKNOWN")
        self.assertIn("validator reported", stored["validation"][0]["user_message"])
        self.assertEqual(result["history_summary"]["source_episode_ids"], ["episode-history-1"])


if __name__ == "__main__":
    unittest.main()
