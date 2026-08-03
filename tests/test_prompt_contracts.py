from __future__ import annotations

import unittest
from pathlib import Path


class PromptContractTests(unittest.TestCase):
    def test_planner_requires_all_compound_task_goals(self) -> None:
        prompt = Path(
            "src/prefmem/agents/prompt/planner/planner-prompt-v2.md"
        ).read_text(encoding="utf-8")

        self.assertIn("completeness audit", prompt)
        self.assertIn("clear and clean the table", prompt)
        self.assertIn("IS_CLEAN(table)", prompt)
        self.assertIn("Never\n  silently omit", prompt)

    def test_hri_rejects_partial_planner_results(self) -> None:
        prompt = Path(
            "src/prefmem/agents/prompt/hri/hri-prompt-v5.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Compare every Planner result", prompt)
        self.assertIn("never present or execute a partial plan", prompt)


if __name__ == "__main__":
    unittest.main()
