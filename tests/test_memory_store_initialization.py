from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prefmem.agents.memory import Memory_Agent


class MemoryStoreInitializationTests(unittest.TestCase):
    def _agent(self, store: Path) -> Memory_Agent:
        agent = object.__new__(Memory_Agent)
        agent.args = SimpleNamespace(memory_store_path=store)
        agent.config = SimpleNamespace(
            embedding_dimensions=768,
            embedding_model="embedding-model",
            embedding_model_base_url="http://127.0.0.1:8080/v1",
            top_k=5,
            top_cap_k=10,
        )
        return agent

    def test_existing_empty_directory_is_a_fresh_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            with patch("prefmem.agents.memory.VLLMEmbeddingGemma"):
                agent.initialize()

        self.assertEqual(agent.pref_json, [])
        self.assertEqual(agent.pref_embedding.shape, (0, 768))

    def test_partial_store_is_rejected_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            (store / "preference.json").write_text("[]", encoding="utf-8")
            agent = self._agent(store)

            with self.assertRaisesRegex(ValueError, "incomplete"):
                agent.initialize()


if __name__ == "__main__":
    unittest.main()
