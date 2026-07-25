from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from memory.models import MemoryContext, MemoryQuery
from simulation.benchmark.ablations import MemoryAblationAgent, apply_memory_mode


class _MemoryAgent:
    def __init__(self) -> None:
        self.history_repository = SimpleNamespace(
            version=3,
            get_summary=lambda _user_id: {"text": "summary"},
        )
        self.preference_repository = SimpleNamespace(version=4)
        self.history_updates = 0
        self.preference_updates = 0
        self.history_retrievals = 0
        self.preference_retrievals = 0

    def get_relevant_history_memory(
        self, _query: MemoryQuery
    ) -> list[dict[str, Any]]:
        self.history_retrievals += 1
        return [{"episode_id": "episode-1"}]

    def get_relevant_preference_memory(
        self, _query: MemoryQuery
    ) -> list[dict[str, Any]]:
        self.preference_retrievals += 1
        return [{"id": "preference-1"}]

    def update_history_memory(self, _conversation: dict[str, Any]) -> dict[str, Any]:
        self.history_updates += 1
        return {"status": "UPDATED"}

    def update_preference_memory(
        self,
        _request: dict[str, Any],
        *,
        consent: Any,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        del consent, transaction_id
        self.preference_updates += 1
        return {"status": "UPDATED"}

    def propose_preference_question(
        self, _request: dict[str, Any]
    ) -> dict[str, Any]:
        return {"question": "Save it?"}


class MemoryAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = _MemoryAgent()
        self.query = MemoryQuery(user_id="participant", request="request")

    def test_no_memory_filters_context_and_blocks_writes(self) -> None:
        agent = MemoryAblationAgent(self.base, "no-memory")

        context = agent.get_memory_context(self.query)
        history = agent.update_history_memory({})
        preference = agent.update_preference_memory({}, consent=object())

        self.assertEqual(context.history_summary, "")
        self.assertEqual(context.relevant_history, [])
        self.assertEqual(context.relevant_preferences, [])
        self.assertEqual(context.history_version, 3)
        self.assertEqual(context.preference_version, 4)
        self.assertEqual(history["status"], "DISABLED_FOR_ABLATION")
        self.assertFalse(preference["committed"])
        self.assertEqual(self.base.history_updates, 0)
        self.assertEqual(self.base.preference_updates, 0)
        self.assertEqual(self.base.history_retrievals, 0)
        self.assertEqual(self.base.preference_retrievals, 0)
        self.assertIsNone(agent.propose_preference_question({}))

    def test_history_only_preserves_history_and_blocks_preferences(self) -> None:
        agent = MemoryAblationAgent(self.base, "history-only")

        context = agent.get_memory_context(self.query)
        agent.update_history_memory({})
        result = agent.update_preference_memory({}, consent=object())

        self.assertEqual(context.history_summary, "summary")
        self.assertTrue(context.relevant_history)
        self.assertEqual(context.relevant_preferences, [])
        self.assertEqual(self.base.history_retrievals, 1)
        self.assertEqual(self.base.preference_retrievals, 0)
        self.assertEqual(self.base.history_updates, 1)
        self.assertFalse(result["committed"])

    def test_preference_only_preserves_preferences_and_blocks_history(self) -> None:
        agent = MemoryAblationAgent(self.base, "preference-only")

        context = agent.get_memory_context(self.query)
        history = agent.update_history_memory({})
        result = agent.update_preference_memory({}, consent=object())

        self.assertEqual(context.history_summary, "")
        self.assertEqual(context.relevant_history, [])
        self.assertTrue(context.relevant_preferences)
        self.assertEqual(self.base.history_retrievals, 0)
        self.assertEqual(self.base.preference_retrievals, 1)
        self.assertEqual(history["status"], "DISABLED_FOR_ABLATION")
        self.assertEqual(result["status"], "UPDATED")
        self.assertEqual(self.base.preference_updates, 1)
        self.assertIsNone(agent.propose_preference_question({}))

    def test_full_mode_returns_original_agent(self) -> None:
        self.assertIs(apply_memory_mode(self.base, "full"), self.base)

    def test_apply_memory_mode_is_idempotent_and_can_replace_an_adapter(self) -> None:
        history_only = apply_memory_mode(self.base, "history-only")

        self.assertIs(
            apply_memory_mode(history_only, "history-only"),
            history_only,
        )
        self.assertIs(apply_memory_mode(history_only, "full"), self.base)
        preference_only = apply_memory_mode(
            history_only,
            "preference-only",
        )
        self.assertIs(preference_only.agent, self.base)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported memory mode"):
            MemoryAblationAgent(self.base, "keyword-magic")


if __name__ == "__main__":
    unittest.main()
