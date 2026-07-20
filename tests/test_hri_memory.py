import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

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
            "origin": "open_preference_answer",
            "independent": True,
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
        self.planner_class = self.planner_patcher.start()
        self.validator_class = self.validator_patcher.start()

        self.memory_agent = memory_class.return_value
        self.planner_agent = self.planner_class.return_value
        self.validator_agent = self.validator_class.return_value
        self.memory_agent.propose_updates.return_value = [{"action": "NOOP"}]
        self.agent = HRI_Agent(config)
        self.agent.prompt_for_dataset_switch = Mock()
        self.planner_agent.plan.return_value = json.dumps(
            {
                "planning_status": "READY",
                "task_complete": False,
                "subtasks": [{"task_instruction": "Place the red cube on the green cube."}],
                "planner_confidence": 0.9,
            }
        )
        self.validator_agent.validate.return_value = json.dumps(
            {
                "outcome": "SUCCESS",
                "task_complete": True,
                "discrepancies": [],
                "validator_confidence": 0.9,
            }
        )

    def tearDown(self):
        self.memory_patcher.stop()
        self.planner_patcher.stop()
        self.validator_patcher.stop()
        self.temp_dir.cleanup()

    def test_new_command_injects_matching_memory(self):
        operation = memory_operation()
        operation["evidence"]["scope_marker"] = "explicit_durable"
        operation["evidence"]["origin"] = "explicit_preference"
        operation["evidence"]["quote"] = "I prefer red, green, then blue for block stacks."
        self.agent.memory_store.apply_operations([operation])

        self.agent._append_user_turn("Stack the blocks", new_command=True)

        payload = json.loads(self.agent.conversation[-1]["content"])
        self.assertEqual(payload["user_message"], "Stack the blocks")
        self.assertEqual(len(payload["memory"]), 1)
        self.assertEqual(payload["memory"][0]["status"], "durable")
        self.assertIn("images", self.agent.conversation[-1])
        self.assertIsInstance(self.agent.conversation[-1]["images"][0], bytes)

    def test_new_command_discards_previous_dialogue_and_images(self):
        self.agent._append_user_turn("First command", new_command=True)
        self.agent.conversation.append({"role": "assistant", "content": "First response"})
        self.agent.current_interaction.append({"role": "assistant", "content": "First response"})

        self.agent._append_user_turn("Second command", new_command=True)

        self.assertEqual([message["role"] for message in self.agent.conversation], ["system", "user"])
        self.assertNotIn("First command", str(self.agent.conversation))
        self.assertEqual(self.agent.current_interaction, [{"role": "user", "content": "Second command"}])
        self.assertEqual(
            sum("images" in message for message in self.agent.conversation),
            1,
        )

    def test_clarification_keeps_command_history_without_duplicating_image(self):
        self.agent._append_user_turn("Stack the blocks", new_command=True)
        assistant_turn = {"role": "assistant", "content": "Which order?"}
        self.agent.conversation.append(assistant_turn)
        self.agent.current_interaction.append(assistant_turn)

        self.agent._append_user_turn("Red, green, blue", new_command=False)

        self.assertEqual(
            [message["role"] for message in self.agent.conversation],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            sum("images" in message for message in self.agent.conversation),
            1,
        )

    def test_resize_configuration_propagates_to_visual_subagents(self):
        planner_config = self.planner_class.call_args.args[0]
        validator_config = self.validator_class.call_args.args[0]

        for visual_config in (planner_config, validator_config):
            self.assertTrue(visual_config.resize_images)
            self.assertEqual((visual_config.image_width, visual_config.image_height), (640, 480))
            self.assertEqual(visual_config.image_jpeg_quality, 85)

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
        self.assertEqual(self.agent.last_task_result["outcome"], "SUCCESS")
        self.agent.prompt_for_dataset_switch.assert_not_called()

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
        self.assertEqual(self.agent.last_task_result["outcome"], "SUCCESS")
        self.agent.prompt_for_dataset_switch.assert_not_called()

    def test_blocked_plan_does_not_call_validator(self):
        self.planner_agent.plan.return_value = json.dumps(
            {
                "planning_status": "BLOCKED",
                "task_complete": False,
                "subtasks": [],
                "planner_confidence": 0.95,
                "failure": {
                    "code": "MISSING_REQUIRED_OBJECT",
                    "expected": "Visible blocks.",
                    "observed": "No blocks visible.",
                    "recoverability": "USER_ASSIST",
                },
            }
        )
        output = json.dumps(
            {"trace": {"mode": "EXECUTE", "confirmed_intent": "Stack the visible blocks."}}
        )

        self.agent.dispatch_to_planner(output)

        self.validator_agent.validate.assert_not_called()
        self.assertEqual(self.agent.last_task_result["outcome"], "BLOCKED")
        self.assertEqual(self.agent.last_task_result["next_action"], "USER_ASSIST")

    def test_repeated_candidate_requires_dedicated_memory_confirmation(self):
        self.memory_agent.propose_updates.return_value = [memory_operation()]
        self.agent.current_interaction = [
            {"role": "user", "content": "Stack the blocks."},
            {"role": "assistant", "content": "Which order would you like?"},
            {"role": "user", "content": "Red, green, blue."},
        ]
        self.agent.update_memory()
        self.agent.update_memory()

        self.assertIsNotNone(self.agent.pending_memory_confirmation)
        candidate = self.agent.memory_store.list_preferences()[0]
        self.assertEqual(candidate["status"], "candidate")

        result = self.agent.answer_memory_confirmation("Yes")

        self.assertEqual(result["action"], "CONFIRMED")
        self.assertEqual(self.agent.memory_store.list_preferences()[0]["status"], "durable")

    def test_report_mode_does_not_plan_or_update_memory(self):
        output = json.dumps(
            {
                "trace": {
                    "mode": "REPORT",
                    "confirmed_intent": None,
                    "failure_code": "MISSING_REQUIRED_OBJECT",
                    "report_reason": "No blocks are visible.",
                }
            }
        )

        mode = self.agent.dispatch_to_planner(output)

        self.assertEqual(mode, "REPORT")
        self.planner_agent.plan.assert_not_called()
        self.memory_agent.propose_updates.assert_not_called()
        self.assertEqual(self.agent.last_task_result["outcome"], "BLOCKED")

    def test_user_cancellation_is_not_reported_as_task_failure(self):
        output = json.dumps(
            {
                "trace": {
                    "mode": "REPORT",
                    "confirmed_intent": None,
                    "failure_code": "USER_CANCELLED",
                    "report_reason": "The user cancelled the task.",
                }
            }
        )
        self.agent.dispatch_to_planner(output)
        self.assertEqual(self.agent.last_task_result["outcome"], "CANCELLED")
        self.assertEqual(self.agent.last_task_result["next_action"], "NONE")

    def test_planner_exception_becomes_unknown_without_validation(self):
        self.planner_agent.plan.side_effect = RuntimeError("planner timeout")
        output = json.dumps(
            {"trace": {"mode": "EXECUTE", "confirmed_intent": "Stack the blocks."}}
        )
        self.agent.dispatch_to_planner(output)
        self.validator_agent.validate.assert_not_called()
        self.assertEqual(self.agent.last_task_result["outcome"], "UNKNOWN")
        self.assertEqual(self.agent.last_task_result["failure"]["code"], "PLANNER_UNAVAILABLE")

    def test_validator_exception_never_becomes_success(self):
        self.validator_agent.validate.side_effect = RuntimeError("camera timeout")
        output = json.dumps(
            {"trace": {"mode": "EXECUTE", "confirmed_intent": "Stack the blocks."}}
        )
        self.agent.dispatch_to_planner(output)
        self.assertEqual(self.agent.last_task_result["outcome"], "UNKNOWN")
        self.assertEqual(self.agent.last_task_result["next_action"], "REOBSERVE")
        self.assertEqual(self.agent.last_task_result["failure"]["code"], "VALIDATOR_UNAVAILABLE")

    def test_prompt_switches_dataset_for_next_command(self):
        with patch("builtins.input", return_value="dataset/v5"):
            switched = HRI_Agent.prompt_for_dataset_switch(self.agent)

        self.assertTrue(switched)
        self.assertEqual(Path(self.agent.initial_frame_path).name, "1.jpg")
        self.assertEqual(Path(self.agent.final_frame_path).name, "3_2.jpg")

        self.agent._append_user_turn("Place the produce into the plates.", new_command=True)
        image_bytes = self.agent.conversation[-1]["images"][0]
        with Image.open(BytesIO(image_bytes)) as image:
            self.assertEqual(image.size, (640, 480))

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
