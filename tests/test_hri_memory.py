import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.hri import HRI_Agent, HRI_Agent_Config


def memory_operation():
    return {
        "action": "UPSERT",
        "preference": {
            "task_type": "stack_blocks",
            "key": "bottom_to_top_color_order",
            "value": ["red", "green", "blue"],
            "context": {"object_type": "cube"},
            "scope": "contextual",
            "confidence": 0.8,
        },
        "evidence": {
            "quote": "First red, then green, then blue.",
            "reason": "The user selected an order after clarification.",
            "scope_marker": "unmarked",
        },
    }


class HRIMemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        config = HRI_Agent_Config()
        config.memory_store_path = str(Path(self.temp_dir.name) / "preferences.json")

        self.memory_patcher = patch("agents.hri.Memory_Agent")
        self.planner_patcher = patch("agents.hri.Planner_Agent")
        self.validator_patcher = patch("agents.hri.Validator_Agent")
        memory_class = self.memory_patcher.start()
        planner_class = self.planner_patcher.start()
        validator_class = self.validator_patcher.start()

        self.memory_agent = memory_class.return_value
        self.planner_agent = planner_class.return_value
        self.validator_agent = validator_class.return_value
        self.agent = HRI_Agent(config)
        self.agent.prompt_for_dataset_switch = Mock()

    def tearDown(self):
        self.memory_patcher.stop()
        self.planner_patcher.stop()
        self.validator_patcher.stop()
        self.temp_dir.cleanup()

    def test_new_command_injects_matching_memory(self):
        operation = memory_operation()
        operation["evidence"]["scope_marker"] = "explicit_durable"
        self.agent.memory_store.apply_operations([operation])

        self.agent._append_user_turn("Stack the blocks", new_command=True)

        payload = json.loads(self.agent.conversation[-1]["content"])
        self.assertEqual(payload["user_message"], "Stack the blocks")
        self.assertEqual(len(payload["memory"]), 1)
        self.assertEqual(payload["memory"][0]["status"], "durable")
        self.assertIn("images", self.agent.conversation[-1])

    def test_execute_curates_then_dispatches_confirmed_intent(self):
        self.memory_agent.propose_updates.return_value = [memory_operation()]
        self.agent.current_interaction = [
            {"role": "user", "content": "Stack the blocks."},
            {"role": "assistant", "content": "Which order?"},
            {"role": "user", "content": "First red, then green, then blue."},
        ]
        confirmed_intent = "Stack red, green, and blue from bottom to top."
        output = json.dumps(
            {
                "trace": {
                    "mode": "EXECUTE",
                    "confirmed_intent": confirmed_intent,
                }
            }
        )

        mode = self.agent.dispatch_to_planner(output)

        self.assertEqual(mode, "EXECUTE")
        self.assertEqual(self.agent.memory_store.list_preferences()[0]["status"], "candidate")
        self.memory_agent.propose_updates.assert_called_once_with(self.agent.current_interaction, [])
        self.planner_agent.plan.assert_called_once_with(confirmed_intent, self.agent.initial_frame_path)
        self.validator_agent.validate.assert_called_once_with(confirmed_intent, self.agent.final_frame_path)
        self.agent.prompt_for_dataset_switch.assert_called_once_with()

    def test_memory_failure_does_not_block_planning(self):
        self.memory_agent.propose_updates.side_effect = RuntimeError("curator unavailable")
        confirmed_intent = "Place the red cube on the green cube."
        output = json.dumps(
            {
                "trace": {
                    "mode": "EXECUTE",
                    "confirmed_intent": confirmed_intent,
                }
            }
        )

        self.agent.dispatch_to_planner(output)

        self.planner_agent.plan.assert_called_once_with(confirmed_intent, self.agent.initial_frame_path)
        self.validator_agent.validate.assert_called_once_with(confirmed_intent, self.agent.final_frame_path)
        self.agent.prompt_for_dataset_switch.assert_called_once_with()

    def test_prompt_switches_dataset_for_next_command(self):
        with patch("builtins.input", return_value="dataset/v5"):
            switched = HRI_Agent.prompt_for_dataset_switch(self.agent)

        self.assertTrue(switched)
        self.assertEqual(Path(self.agent.initial_frame_path).name, "1.jpg")
        self.assertEqual(Path(self.agent.final_frame_path).name, "3_2.jpg")

        self.agent._append_user_turn("Place the produce into the plates.", new_command=True)
        self.assertEqual(self.agent.conversation[-1]["images"], [self.agent.initial_frame_path])

    def test_prompt_enter_keeps_current_dataset(self):
        original_episode = self.agent.dataset_episode

        with patch("builtins.input", return_value=""):
            switched = HRI_Agent.prompt_for_dataset_switch(self.agent)

        self.assertFalse(switched)
        self.assertIs(self.agent.dataset_episode, original_episode)

    def test_prompt_invalid_path_keeps_current_dataset(self):
        original_episode = self.agent.dataset_episode

        with patch("builtins.input", return_value="dataset/missing"):
            switched = HRI_Agent.prompt_for_dataset_switch(self.agent)

        self.assertFalse(switched)
        self.assertIs(self.agent.dataset_episode, original_episode)


if __name__ == "__main__":
    unittest.main()