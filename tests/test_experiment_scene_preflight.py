import unittest

from experiments.harness.scene_preflight import (
    ScenePreflightError,
    trigger_action_kind,
)
from experiments.harness.scenarios import load_scenarios


class ScenePreflightSchemaTests(unittest.TestCase):
    def test_every_frozen_trigger_action_has_exact_handler_class(self) -> None:
        actions = {
            trigger.action
            for scenario in load_scenarios().scenarios
            for trigger in scenario.triggers
        }
        self.assertGreaterEqual(len(actions), 50)
        for action in actions:
            with self.subTest(action=action):
                self.assertIn(
                    trigger_action_kind(action),
                    {
                        "scripted_input",
                        "memory",
                        "scene_move",
                        "scene_insert",
                        "scene_remove",
                        "visual_evidence",
                        "model_fault",
                        "execution_fault",
                        "host_fence",
                    },
                )

    def test_unknown_or_ambiguous_action_fails_closed(self) -> None:
        with self.assertRaises(ScenePreflightError):
            trigger_action_kind("do something unspecified")


if __name__ == "__main__":
    unittest.main()
