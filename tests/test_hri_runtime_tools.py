from __future__ import annotations

import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from prefmem.agents.hri import HRI_Agent


class BindableModel:
    def __init__(self) -> None:
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self


class RecordingModel:
    def __init__(self, response=None) -> None:
        self.calls = []
        self.response = response

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return (
            self.response
            if self.response is not None
            else AIMessage(content="ok")
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.shutdown_event = threading.Event()
        self.notifications = queue.SimpleQueue()
        self.calls = []
        self.timeout_checks = 0

    def request_goal_preview(self, **kwargs):
        self.calls.append(("request_goal_preview", kwargs))
        return {"goal_contract": {"goal_id": "goal-1", "revision": 1}}

    def confirm_goal(self, **kwargs):
        self.calls.append(("confirm_goal", kwargs))
        return {"state": "PLANNING"}

    def resume_current_task(self):
        self.calls.append(("resume_current_task", {}))
        return {"state": "EXECUTING"}

    def request_replan(self, **kwargs):
        self.calls.append(("request_replan", kwargs))
        return {"state": "PLANNING"}

    def context_json(self):
        return '{"state":"EXECUTING","cycle_id":2}'

    def check_timeout(self):
        self.timeout_checks += 1
        return False


class HRIRuntimeToolsTests(unittest.TestCase):
    def make_attached_hri(self):
        hri = object.__new__(HRI_Agent)
        hri._hri_model = BindableModel()
        hri.runtime = None
        runtime = FakeRuntime()
        hri.attach_runtime(runtime)
        return hri, runtime

    def test_attach_runtime_replaces_generic_tool_with_narrow_surface(self):
        hri, _ = self.make_attached_hri()

        self.assertEqual(
            set(hri.TOOLS_BY_NAME),
            {
                "call_memory_agent",
                "request_goal_preview",
                "confirm_goal_execution",
                "resume_current_task",
                "request_execution_replan",
            },
        )
        self.assertNotIn("call_sub_agent", hri.TOOLS_BY_NAME)

    def test_runtime_wrappers_forward_exact_confirmation_and_guidance(self):
        hri, runtime = self.make_attached_hri()

        preview = json.loads(
            hri.request_goal_preview(
                "put the red block in the tray",
                constraints=["keep the blue block still"],
            )
        )
        confirmed = json.loads(
            hri.confirm_goal_execution("goal-1", 1, True)
        )
        replanned = json.loads(
            hri.request_execution_replan("approach from the left")
        )

        self.assertEqual(preview["goal_contract"]["goal_id"], "goal-1")
        self.assertEqual(confirmed["state"], "PLANNING")
        self.assertEqual(replanned["state"], "PLANNING")
        self.assertEqual(
            runtime.calls,
            [
                (
                    "request_goal_preview",
                    {
                        "clarified_goal": "put the red block in the tray",
                        "constraints": ("keep the blue block still",),
                        "operator_guidance": None,
                    },
                ),
                (
                    "confirm_goal",
                    {
                        "goal_id": "goal-1",
                        "revision": 1,
                        "confirmed": True,
                    },
                ),
                (
                    "request_replan",
                    {"operator_guidance": "approach from the left"},
                ),
            ],
        )

    def test_goal_preview_tool_normalizes_string_constraints(self):
        hri, runtime = self.make_attached_hri()
        constraint_text = (
            "Trash must go in the bin. Electronics must go in the box."
        )

        output = json.loads(
            hri.request_goal_preview_tool.invoke(
                {
                    "clarified_goal": "Tidy up the items on the floor.",
                    "constraints": constraint_text,
                }
            )
        )

        self.assertEqual(output["goal_contract"]["goal_id"], "goal-1")
        self.assertEqual(
            runtime.calls,
            [
                (
                    "request_goal_preview",
                    {
                        "clarified_goal": "Tidy up the items on the floor.",
                        "constraints": (constraint_text,),
                        "operator_guidance": None,
                    },
                )
            ],
        )
        constraints_schema = (
            hri.request_goal_preview_tool.args_schema.model_json_schema()[
                "properties"
            ]["constraints"]
        )
        self.assertEqual(constraints_schema["type"], "array")

    def test_direct_goal_preview_preserves_string_as_one_constraint(self):
        hri, runtime = self.make_attached_hri()
        constraint_text = "Keep batteries away from the trash bin."

        hri.request_goal_preview(
            "Tidy up the items.",
            constraints=constraint_text,
        )

        self.assertEqual(
            runtime.calls[0][1]["constraints"],
            (constraint_text,),
        )

    def test_invalid_tool_arguments_return_error_instead_of_crashing(self):
        hri, runtime = self.make_attached_hri()
        tool_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "request_goal_preview",
                    "args": {
                        "clarified_goal": "Tidy up.",
                        "constraints": {"invalid": "object"},
                    },
                    "id": "call-invalid-constraints",
                    "type": "tool_call",
                }
            ],
        )

        update = hri.tool_node({"messages": [tool_request]})
        error = json.loads(update["messages"][0].content)

        self.assertEqual(error["status"], "ERROR")
        self.assertIn("ValidationError", error["error"])
        self.assertEqual(runtime.calls, [])

    def test_confirmation_tool_does_not_coerce_execution_authority(self):
        malformed_arguments = (
            {"goal_id": "goal-1", "revision": "1", "confirmed": True},
            {"goal_id": "goal-1", "revision": 1, "confirmed": "true"},
            {"goal_id": "goal-1", "revision": 1, "confirmed": 1},
        )

        for index, arguments in enumerate(malformed_arguments):
            with self.subTest(arguments=arguments):
                hri, runtime = self.make_attached_hri()
                tool_request = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "confirm_goal_execution",
                            "args": arguments,
                            "id": f"call-malformed-confirm-{index}",
                            "type": "tool_call",
                        }
                    ],
                )

                update = hri.tool_node({"messages": [tool_request]})
                error = json.loads(update["messages"][0].content)

                self.assertEqual(error["status"], "ERROR")
                self.assertIn("ValidationError", error["error"])
                self.assertEqual(runtime.calls, [])

    def test_goal_preview_rejects_misspelled_constraint_field(self):
        hri, runtime = self.make_attached_hri()
        tool_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "request_goal_preview",
                    "args": {
                        "clarified_goal": "Tidy up.",
                        "constraint": "Keep batteries out of the bin.",
                    },
                    "id": "call-misspelled-constraint",
                    "type": "tool_call",
                }
            ],
        )

        update = hri.tool_node({"messages": [tool_request]})
        error = json.loads(update["messages"][0].content)

        self.assertEqual(error["status"], "ERROR")
        self.assertIn("extra_forbidden", error["error"])
        self.assertEqual(runtime.calls, [])

    def test_missing_tool_call_id_gets_safe_fallback(self):
        hri, runtime = self.make_attached_hri()
        tool_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "request_goal_preview",
                    "args": {"clarified_goal": "Tidy up.", "constraints": []},
                    "id": None,
                    "type": "tool_call",
                }
            ],
        )

        update = hri.tool_node({"messages": [tool_request]})

        self.assertEqual(
            update["messages"][0].tool_call_id,
            "prefmem-tool-call-0",
        )
        self.assertEqual(len(runtime.calls), 1)

    def test_invalid_protocol_tool_call_has_visible_safe_response(self):
        invalid_response = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "request_goal_preview",
                    "args": "{bad",
                    "id": "invalid-call",
                    "error": "invalid JSON",
                    "type": "invalid_tool_call",
                }
            ],
        )
        model = RecordingModel(invalid_response)
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="system")
        hri.system_prompt = "system"
        hri.hri_llm = model
        hri.metrics = None
        hri.thinking_enabled = False
        hri.runtime = None

        update = hri.llm_call(
            {"messages": [HumanMessage(content="tidy up")], "llm_calls": 0},
            SimpleNamespace(context={"current_frame": {}}),
        )

        self.assertIn("nothing was executed", update["messages"][0].content)

    def test_live_runtime_state_is_injected_into_each_model_call(self):
        model = RecordingModel()
        runtime = FakeRuntime()
        hri = object.__new__(HRI_Agent)
        hri.config = SimpleNamespace(system_prompt="system")
        hri.system_prompt = "system"
        hri.hri_llm = model
        hri.metrics = None
        hri.thinking_enabled = False
        hri.runtime = runtime

        hri.llm_call(
            {"messages": [HumanMessage(content="status?")], "llm_calls": 0},
            SimpleNamespace(context={"current_frame": {}}),
        )

        system = next(
            message.content
            for message in model.calls[-1]
            if isinstance(message, SystemMessage)
        )
        self.assertIn("PREFMEM_RUNTIME_STATE", system)
        self.assertIn('"state":"EXECUTING"', system)
        self.assertIn('"cycle_id":2', system)

    def test_console_wait_polls_timeout_and_exits_on_emergency(self):
        hri, runtime = self.make_attached_hri()

        with patch("builtins.input", return_value="continue"):
            self.assertEqual(hri._read_user_input_interruptibly(), "continue")
        self.assertGreaterEqual(runtime.timeout_checks, 1)

        runtime.shutdown_event.set()
        with patch("builtins.input", return_value="ignored"):
            self.assertIsNone(hri._read_user_input_interruptibly())


if __name__ == "__main__":
    unittest.main()
