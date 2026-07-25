from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main as main_module
from agents.configs import PrefMemConfig
from agents.contracts import ExecutionResult
from agents.hri import HRIOrchestrator
from agents.planner import PlannerAgent
from agents.validator import ValidatorAgent
from memory.models import MemoryContext
from tests.fakes import RecordingExecutor, RecordingMemoryAgent, ScriptedJsonModel


ROOT = Path(__file__).resolve().parents[1]


class DisplayAllParserTests(unittest.TestCase):
    def test_display_all_is_opt_in(self) -> None:
        parser = main_module.build_parser()

        self.assertFalse(parser.parse_args([]).display_all)
        self.assertTrue(parser.parse_args(["--display_all"]).display_all)

    def test_main_forwards_display_all_to_runtime_without_persisting_it_in_agent_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "terminal.txt"
            prefmem_instance = Mock()
            argv = [
                "main.py",
                "--display_all",
                "--transcript",
                str(transcript),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(main_module, "PrefMem", return_value=prefmem_instance) as prefmem,
            ):
                main_module.main()

            prefmem.assert_called_once()
            config = prefmem.call_args.args[0]
            self.assertIsInstance(config, PrefMemConfig)
            self.assertFalse(
                hasattr(config, "display_all"),
                "Terminal verbosity is UI state, not durable agent behavior.",
            )
            prefmem_instance.start.assert_called_once_with(display_all=True)

    def test_prefmem_start_forwards_display_setting_to_interactive_hri(self) -> None:
        hri = Mock()
        with (
            patch.object(main_module, "HRIOrchestrator", return_value=hri),
            patch("builtins.print"),
        ):
            runtime = main_module.PrefMem(PrefMemConfig())
            runtime.start(display_all=True)

        hri.get_response.assert_called_once_with(display_all=True)


class DisplayAllRuntimeTests(unittest.TestCase):
    @staticmethod
    def _config(directory: str) -> PrefMemConfig:
        config = PrefMemConfig(
            workspace_root=str(ROOT),
            dataset_path=str(ROOT / "dataset" / "v3"),
            history_store_path=str(Path(directory) / "history.json"),
            history_outbox_path=str(Path(directory) / "history_outbox.json"),
            preference_store_path=str(Path(directory) / "preferences.json"),
            user_id="display-test-user",
            max_replans=0,
        )
        config.hri.model = "offline-only"
        config.planner.model = "offline-only"
        config.validator.model = "offline-only"
        return config

    def _orchestrator(self, directory: str) -> HRIOrchestrator:
        config = self._config(directory)
        hri_model = ScriptedJsonModel(
            {
                "resolve_hri_turn": {
                    "mode": "EXECUTE",
                    "user_message": "I will execute the confirmed RGB stack.",
                    "raw_diagnostic": "HRI_RAW_SENTINEL",
                    "task_contract": {
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "parameters": {"order": "RGB"},
                    },
                    "memory_action": {"action": "NONE"},
                }
            }
        )
        planner_model = ScriptedJsonModel(
            {
                "plan_task": {
                    "planning_status": "READY",
                    "planner_confidence": 0.99,
                    "raw_diagnostic": "PLANNER_RAW_SENTINEL",
                    "preconditions": [],
                    "subtasks": [
                        {"task_instruction": "Place green on red, then blue on green."}
                    ],
                    "validation_spec": {
                        "spec_id": "display-spec",
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "goal_conditions": [
                            {
                                "id": "display-goal",
                                "description": "The stack is RGB bottom-to-top.",
                                "predicate": "STABLE_STACK",
                                "arguments": [
                                    "red block",
                                    "green block",
                                    "blue block",
                                ],
                            }
                        ],
                    },
                }
            }
        )

        def validate(payload: dict[str, object]) -> dict[str, object]:
            spec = payload["validation_spec"]
            assert isinstance(spec, dict)
            goals = spec["goal_conditions"]
            assert isinstance(goals, (list, tuple))
            return {
                "spec_id": spec["spec_id"],
                "outcome": "SUCCESS",
                "task_complete": True,
                "goal_checks": [
                    {"goal_id": goal["id"], "satisfied": True}
                    for goal in goals
                ],
                "discrepancies": [],
                "validator_confidence": 0.99,
                "recoverability": "NONE",
                "user_message": "Task Complete.",
                "raw_diagnostic": "VALIDATOR_RAW_SENTINEL",
            }

        validator_model = ScriptedJsonModel({"validate_task": validate})
        memory = RecordingMemoryAgent(
            [MemoryContext(history_summary="MEMORY_CONTEXT_SENTINEL")]
        )

        class SentinelExecutor(RecordingExecutor):
            def execute(self, plan: object, *, attempt: int = 1) -> ExecutionResult:
                result = super().execute(plan, attempt=attempt)  # type: ignore[arg-type]
                result.evidence["raw_diagnostic"] = "VLA_EXECUTION_SENTINEL"
                return result

        executor = SentinelExecutor(str(ROOT / "dataset" / "v3" / "12.png"))
        return HRIOrchestrator(
            config,
            hri_model=hri_model,
            memory_agent=memory,
            planner_agent=PlannerAgent(
                config.planner, vision=config.vision, model=planner_model
            ),
            validator_agent=ValidatorAgent(
                config.validator, vision=config.vision, model=validator_model
            ),
            executor=executor,
        )

    def _capture_session(self, *, display_all: bool) -> str:
        terminal = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._orchestrator(directory)
            # Command, empty dataset selection, then terminate the next prompt.
            with (
                patch("builtins.input", side_effect=["Stack the blocks", "", EOFError()]),
                patch("sys.stdout", terminal),
            ):
                orchestrator.get_response(display_all=display_all)
        return terminal.getvalue()

    def test_default_terminal_view_remains_hri_only(self) -> None:
        output = self._capture_session(display_all=False)

        self.assertIn("Task Complete.", output)
        for hidden in (
            "[Display All]",
            "HRI_RAW_SENTINEL",
            "MEMORY_CONTEXT_SENTINEL",
            "PLANNER_RAW_SENTINEL",
            "VLA_EXECUTION_SENTINEL",
            "VALIDATOR_RAW_SENTINEL",
        ):
            self.assertNotIn(hidden, output)

    def test_display_all_prints_each_available_agent_and_assurance_payload(self) -> None:
        output = self._capture_session(display_all=True)

        self.assertIn("Task Complete.", output)
        for agent in (
            "HRI",
            "Memory",
            "Planner",
            "VLA",
            "Validator",
            "Task Assurance",
        ):
            self.assertIn(f"[Display All] {agent}", output)
        for visible_payload in (
            "HRI_RAW_SENTINEL",
            "MEMORY_CONTEXT_SENTINEL",
            "PLANNER_RAW_SENTINEL",
            "VLA_EXECUTION_SENTINEL",
            "VALIDATOR_RAW_SENTINEL",
        ):
            self.assertIn(visible_payload, output)

    def test_display_all_includes_rejected_raw_hri_attempt_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            hri_model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": [
                        {
                            "mode": "INVALID_MODE",
                            "user_message": "Malformed attempt.",
                            "raw_diagnostic": "REJECTED_HRI_ATTEMPT_SENTINEL",
                        },
                        {
                            "mode": "ASK",
                            "user_message": "In what order should I stack the blocks?",
                            "pending_question": {
                                "kind": "TASK_CLARIFICATION",
                                "payload": {},
                            },
                            "memory_action": {"action": "NONE"},
                        },
                    ]
                }
            )
            memory = RecordingMemoryAgent()
            executor = RecordingExecutor(str(ROOT / "dataset" / "v3" / "12.png"))
            orchestrator = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=memory,
                planner_agent=Mock(),
                validator_agent=Mock(),
                executor=executor,
            )
            terminal = io.StringIO()
            with (
                patch("builtins.input", side_effect=["Stack the blocks", EOFError()]),
                patch("sys.stdout", terminal),
            ):
                orchestrator.get_response(display_all=True)

        output = terminal.getvalue()
        self.assertIn("[Display All] HRI", output)
        self.assertIn("REJECTED_HRI_ATTEMPT_SENTINEL", output)
        self.assertIn("In what order should I stack the blocks?", output)


if __name__ == "__main__":
    unittest.main()
