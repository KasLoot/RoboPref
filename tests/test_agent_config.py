from __future__ import annotations

import unittest

from prefmem.agents.config import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_EMBEDDING_BASE_URL,
    ControllerConfig,
    EmbeddingConfig,
    HRI_Config,
    Monitor_Config,
)


class AgentConfigTests(unittest.TestCase):
    def test_local_service_defaults_match_documented_ports(self) -> None:
        self.assertEqual(DEFAULT_AGENT_BASE_URL, "http://127.0.0.1:8000/v1")
        self.assertEqual(
            DEFAULT_EMBEDDING_BASE_URL,
            "http://127.0.0.1:8080/v1",
        )
        self.assertEqual(EmbeddingConfig().dimensions, 768)

    def test_prompt_paths_are_independent_of_working_directory(self) -> None:
        hri = HRI_Config("vllm")
        monitor = Monitor_Config("vllm")

        self.assertEqual(hri.system_prompt_path.name, "hri-prompt-v3.md")
        self.assertIn("RETRIEVE_MEMORY", hri.system_prompt)
        self.assertEqual(monitor.system_prompt_path.name, "prompt-v1.md")
        self.assertIn("ON_GOING", monitor.system_prompt)

    def test_cyclic_controller_budgets_are_finite(self) -> None:
        config = ControllerConfig()

        self.assertEqual(config.max_memory_retrievals_per_turn, 1)
        self.assertGreater(config.max_monitor_cycles_per_subtask, 0)
        self.assertGreater(config.max_validation_attempts, 0)
        self.assertGreater(config.max_dispatches_per_episode, 0)

    def test_invalid_controller_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "max_validation_attempts",
        ):
            ControllerConfig(max_validation_attempts=0)


if __name__ == "__main__":
    unittest.main()
