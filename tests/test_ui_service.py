from __future__ import annotations

import queue
import json
import threading
import unittest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from ui.service import AgentConsole


class FakeFrame:
    image_block = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,/9j/"},
    }


class FakeMetrics:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def summary(self, *, turn_seconds: float) -> dict:
        return {"total": {"turn_wall_seconds": turn_seconds}}


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def invoke(self, value, *, config, context):
        self.calls.append((value, config, context))
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "some_tool",
                            "args": {},
                            "id": "tool-1",
                        }
                    ],
                ),
                AIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "private"},
                        {"type": "text", "text": "Visible answer"},
                    ],
                    additional_kwargs={"reasoning": "also private"},
                ),
            ]
        }


class FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


class StreamingGraph(FakeGraph):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls: list[dict] = []

    def invoke(self, value, *, config, context):  # pragma: no cover - guard
        del value, config, context
        raise AssertionError("stream_turn must use the graph streaming API")

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        metadata = {"langgraph_node": "HRI Agent"}
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning": "Recall preference. "},
                ),
                metadata,
            ),
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "call_memory_agent",
                            "args": '{"message":"RETRIEVE ',
                            "id": "memory-1",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                ),
                metadata,
            ),
        }
        tool_call = {
            "name": "call_memory_agent",
            "args": {"message": "RETRIEVE REQUEST: blocks"},
            "id": "memory-1",
        }
        yield {
            "type": "updates",
            "data": {
                "HRI Agent": {
                    "messages": [AIMessage(content="", tool_calls=[tool_call])]
                }
            },
        }
        yield {
            "type": "updates",
            "data": {
                "Tool Node": {
                    "messages": [
                        ToolMessage(
                            content='{"preference":"use the left side"}',
                            tool_call_id="memory-1",
                            name="call_memory_agent",
                        )
                    ]
                }
            },
        }
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content="Use the "), metadata),
        }
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content="left side."), metadata),
        }
        yield {
            "type": "updates",
            "data": {
                "HRI Agent": {
                    "messages": [AIMessage(content="Use the left side.")]
                }
            },
        }


class DraftStreamingGraph(FakeGraph):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def stream(self, **kwargs):
        del kwargs
        metadata = {"langgraph_node": "HRI Agent"}
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content="Partial "), metadata),
        }
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the stream")
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content="answer"), metadata),
        }
        yield {
            "type": "updates",
            "data": {
                "HRI Agent": {
                    "messages": [AIMessage(content="Partial answer")]
                }
            },
        }


class FakeHRI:
    def __init__(self) -> None:
        self.hri_agent = FakeGraph()
        self.hri_agent.checkpointer = FakeCheckpointer()
        self.metrics = FakeMetrics()
        self.completed_memory_mutations = []

    @staticmethod
    def _render_runtime_notification(value, *, authoritative_state):
        del authoritative_state
        if isinstance(value, dict):
            return value.get("summary", str(value))
        return str(value)


class BlockingGraph(FakeGraph):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def invoke(self, value, *, config, context):
        self.calls.append((value, config, context))
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the graph")
        return {"messages": [AIMessage(content="Finished")]}


class HistoricalGraph(FakeGraph):
    def invoke(self, value, *, config, context):
        self.calls.append((value, config, context))
        return {
            "messages": [
                HumanMessage(content="Earlier question"),
                AIMessage(content="Earlier answer"),
                HumanMessage(content="Current question"),
                AIMessage(content=""),
            ]
        }


class FakeService:
    running = False
    active_publication_id = None


class FakeEmergency:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.latched = False
        self.calls = []

    def trigger(self, reason, *, publication_id=None):
        if self.latched:
            return False
        self.latched = True
        self.calls.append((reason, publication_id))
        self.runtime.state = "EMERGENCY_STOPPED"
        self.runtime.notifications.put(f"EMERGENCY STOP: {reason}")
        return True


class FakeRuntime:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.task_phase = None
        self.notifications = queue.SimpleQueue()
        self.monitor = FakeService()
        self.validator = FakeService()
        self.emergency = FakeEmergency(self)
        self.confirmations = []
        self.replans = []
        self.resume_count = 0
        self.timeout_checks = 0
        self.close_count = 0

    def frame_source(self):
        return FakeFrame()

    def context_dict(self):
        return {
            "state": self.state,
            "current_task": {
                "publication_id": "publication-1",
                "instruction": "Move the block.",
                "phase": self.task_phase,
            },
            "emergency_latched": self.emergency.latched,
        }

    def confirm_goal(self, *, goal_id, revision, confirmed):
        self.confirmations.append((goal_id, revision, confirmed))
        self.state = "PLANNING" if confirmed else "AWAITING_CONFIRMATION"
        return self.context_dict()

    def resume_current_task(self):
        self.resume_count += 1
        self.state = "EXECUTING"
        return self.context_dict()

    def request_replan(self, guidance=None):
        self.replans.append(guidance)
        self.state = "PLANNING"
        return self.context_dict()

    def check_timeout(self):
        self.timeout_checks += 1
        return False

    def close(self):
        self.close_count += 1


class AgentConsoleTests(unittest.TestCase):
    def make_console(self, **kwargs):
        runtime = FakeRuntime()
        hri = FakeHRI()
        calls = []

        def factory():
            calls.append(True)
            return runtime, hri

        console = AgentConsole(runtime_factory=factory, **kwargs)
        return console, runtime, hri, calls

    def test_run_turn_lazily_starts_and_returns_only_visible_text(self):
        console, runtime, hri, factory_calls = self.make_console(
            thread_id="gui-session",
        )

        before = console.snapshot()
        self.assertFalse(before["started"])
        self.assertFalse(before["busy"])

        result = console.run_turn("  Please move the block.  ")

        self.assertTrue(result["ok"])
        self.assertEqual(result["assistant_text"], "Visible answer")
        self.assertNotIn("private", result["assistant_text"])
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(hri.metrics.reset_count, 0)
        _, config, context = hri.hri_agent.calls[0]
        self.assertEqual(
            config["configurable"]["thread_id"],
            "gui-session",
        )
        self.assertEqual(context["current_frame"], FakeFrame.image_block)
        self.assertEqual(result["context"], runtime.context_dict())
        for key in ("ok", "message", "payload", "assistant_text"):
            self.assertIn(key, result)

    def test_stream_turn_uses_langgraph_events_and_builds_safe_live_trace(self):
        console, _runtime, hri, _calls = self.make_console(
            thread_id="gui-session",
        )
        graph = StreamingGraph()
        hri.hri_agent = graph

        events = []
        live = None
        for event in console.stream_turn("Which side should I use?"):
            events.append(event)
            if event["kind"] in {"tool_result", "assistant_delta"}:
                live = console.cached_snapshot()["active_turn"]

        self.assertEqual(events[0]["kind"], "turn_start")
        self.assertEqual(events[-1]["kind"], "turn_complete")
        self.assertEqual(
            [event["sequence"] for event in events],
            sorted(event["sequence"] for event in events),
        )
        self.assertEqual({event["turn_id"] for event in events}, {"turn-1"})
        kinds = [event["kind"] for event in events]
        for kind in (
            "thinking_delta",
            "tool_call_delta",
            "tool_call",
            "tool_result",
            "assistant_delta",
            "turn_complete",
        ):
            self.assertIn(kind, kinds)
        result = events[-1]["payload"]["result"]
        self.assertTrue(result["ok"])
        self.assertEqual(result["assistant_text"], "Use the left side.")

        call = graph.stream_calls[0]
        self.assertEqual(call["version"], "v2")
        self.assertEqual(call["stream_mode"], ["messages", "updates"])
        self.assertEqual(
            call["config"]["configurable"]["thread_id"], "gui-session"
        )
        self.assertIsInstance(call["input"]["messages"][0], HumanMessage)
        self.assertNotIn("image_url", json.dumps(events))

        snapshot = console.cached_snapshot()
        self.assertIsNotNone(live)
        self.assertEqual(live["status"], "streaming")
        self.assertEqual(live["assistant_text"], "Use the left side.")
        self.assertEqual(live["reasoning_text"], "Recall preference. ")
        self.assertEqual(live["tools"][0]["status"], "complete")
        self.assertIn("left side", live["tools"][0]["result"])
        self.assertIsNone(snapshot["active_turn"])
        self.assertEqual(snapshot["agents"]["memory"]["status"], "idle")
        self.assertEqual(snapshot["trace"], events)
        self.assertEqual(
            {message["turn_id"] for message in snapshot["transcript"]},
            {"turn-1"},
        )
        json.dumps(snapshot["trace"])

    def test_active_turn_exposes_non_consuming_partial_assistant_draft(self):
        console, _runtime, hri, _calls = self.make_console()
        graph = DraftStreamingGraph()
        hri.hri_agent = graph
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Stream this")),
        )

        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))
        first = console.cached_snapshot()
        second = console.cached_snapshot()

        self.assertEqual(first["active_turn"], second["active_turn"])
        self.assertEqual(first["active_turn"]["status"], "streaming")
        self.assertEqual(first["active_turn"]["assistant_text"], "Partial ")
        self.assertIsNone(first["active_turn"]["finished_at"])
        self.assertEqual(first["agents"]["hri"]["status"], "working")
        self.assertEqual(first["agents"]["hri"]["activity"], "Responding")

        graph.release.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["ok"])
        finished = console.cached_snapshot()
        self.assertIsNone(finished["active_turn"])
        self.assertEqual(finished["transcript"][-1]["content"], "Partial answer")

    def test_trace_history_is_bounded_by_capacity(self):
        console, _runtime, _hri, _calls = self.make_console(trace_capacity=3)

        result = console.run_turn("Bound this trace")
        snapshot = console.cached_snapshot()

        self.assertTrue(result["ok"])
        self.assertEqual(len(snapshot["trace"]), 3)
        self.assertGreater(snapshot["trace_cursor"], len(snapshot["trace"]))
        self.assertEqual(snapshot["trace"][-1]["kind"], "turn_complete")

    def test_reset_chat_rotates_checkpoint_without_touching_runtime_state(self):
        console, runtime, hri, factory_calls = self.make_console(
            thread_id="gui-session"
        )
        self.assertTrue(console.run_turn("First turn")["ok"])
        runtime.state = "EXECUTING"
        hri.completed_memory_mutations.extend([{"mutation_id": "m-1"}])
        runtime_identity = console.runtime
        old_thread_id = console.thread_id
        old_event_cursor = console.cached_snapshot()["event_cursor"]

        reset = console.reset_chat_session()

        self.assertTrue(reset["ok"])
        self.assertEqual(reset["payload"]["old_thread_id"], old_thread_id)
        self.assertNotEqual(reset["payload"]["new_thread_id"], old_thread_id)
        self.assertTrue(reset["payload"]["new_thread_id"].startswith("gui-session-"))
        self.assertIs(console.runtime, runtime_identity)
        self.assertEqual(runtime.state, "EXECUTING")
        self.assertEqual(runtime.close_count, 0)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(hri.completed_memory_mutations, [])
        self.assertEqual(
            hri.hri_agent.checkpointer.deleted_threads, [old_thread_id]
        )
        snapshot = console.cached_snapshot()
        self.assertEqual(snapshot["transcript"], [])
        self.assertEqual(snapshot["trace"], [])
        self.assertIsNone(snapshot["active_turn"])
        self.assertGreater(snapshot["event_cursor"], old_event_cursor)

        self.assertTrue(console.run_turn("Second turn")["ok"])
        _, config, _context = hri.hri_agent.calls[-1]
        self.assertEqual(
            config["configurable"]["thread_id"], console.thread_id
        )

    def test_reset_chat_is_rejected_while_a_turn_is_in_flight(self):
        console, _runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Wait")),
        )
        old_thread_id = console.thread_id

        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))
        reset = console.reset_chat_session()

        self.assertFalse(reset["ok"])
        self.assertIn("HRI turn", reset["message"])
        self.assertEqual(console.thread_id, old_thread_id)
        graph.release.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["ok"])

    def test_reset_cannot_receive_a_stale_terminal_event_after_rotation(self):
        console, _runtime, _hri, _calls = self.make_console(
            thread_id="gui-session"
        )
        operation_released = threading.Event()
        allow_turn_to_return = threading.Event()
        original_end_operation = console._end_operation
        result = {}

        def gated_end_operation():
            original_end_operation()
            if threading.current_thread() is worker:
                operation_released.set()
                if not allow_turn_to_return.wait(timeout=2.0):
                    raise TimeoutError("test did not release the completed turn")

        console._end_operation = gated_end_operation
        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Finish old chat")),
        )

        worker.start()
        self.assertTrue(operation_released.wait(timeout=1.0))
        reset = console.reset_chat_session()

        self.assertTrue(reset["ok"])
        self.assertEqual(console.cached_snapshot()["trace"], [])
        allow_turn_to_return.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["ok"])
        snapshot = console.cached_snapshot()
        self.assertEqual(snapshot["trace"], [])
        self.assertEqual(snapshot["transcript"], [])

    def test_cached_snapshot_does_not_poll_runtime_context(self):
        console, runtime, _hri, _calls = self.make_console()
        context_calls = 0
        original_context = runtime.context_dict

        def counted_context():
            nonlocal context_calls
            context_calls += 1
            return original_context()

        runtime.context_dict = counted_context
        console.start()
        calls_after_start = context_calls

        first = console.cached_snapshot()
        second = console.cached_snapshot()

        self.assertEqual(context_calls, calls_after_start)
        self.assertEqual(first["context"], original_context())
        self.assertEqual(first, second)

    def test_worker_threads_need_matching_publication_and_phase_to_be_active(self):
        console, runtime, _hri, _calls = self.make_console()
        console.start()
        runtime.monitor.running = True
        runtime.monitor.active_publication_id = "publication-1"
        runtime.validator.running = True
        runtime.validator.active_publication_id = "publication-1"

        # A live worker thread alone is just standby.  The current controller
        # publication and phase determine which service is doing agent work.
        runtime.state = "EXECUTING"
        runtime.task_phase = "STEP"
        executing = console.snapshot()["agents"]
        self.assertEqual(executing["monitor"]["status"], "watching")
        self.assertEqual(executing["validator"]["status"], "idle")

        runtime.state = "FINAL_VALIDATION"
        runtime.task_phase = "FINAL_VALIDATION"
        validating = console.snapshot()["agents"]
        self.assertEqual(validating["monitor"]["status"], "idle")
        self.assertEqual(validating["validator"]["status"], "working")

        runtime.validator.active_publication_id = "stale-publication"
        stale = console.snapshot()["agents"]
        self.assertEqual(stale["validator"]["status"], "idle")

    def test_agent_status_mappings_match_real_agent_boundaries(self):
        self.assertEqual(
            AgentConsole._tool_agents("request_goal_preview"),
            ("planner",),
        )
        self.assertEqual(
            AgentConsole._tool_agents("call_memory_agent"),
            ("memory",),
        )
        self.assertEqual(
            AgentConsole._tool_agents(
                "confirm_goal_execution", {"confirmed": False}
            ),
            (),
        )
        self.assertEqual(
            AgentConsole._tool_agents(
                "confirm_goal_execution", {"confirmed": True}
            ),
            ("validator",),
        )
        self.assertEqual(
            AgentConsole._tool_agents("request_execution_replan"),
            ("planner",),
        )
        self.assertEqual(
            AgentConsole._control_agents("goal confirmation"),
            ("validator",),
        )
        self.assertEqual(AgentConsole._control_agents("goal decline"), ())

    def test_planning_context_projects_only_planner_as_working(self):
        console, runtime, _hri, _calls = self.make_console()
        console.start()
        console._set_agent_status(
            "validator", "working", "Goal confirmation"
        )
        runtime.state = "PLANNING"

        agents = console.snapshot()["agents"]

        self.assertEqual(agents["planner"]["status"], "working")
        self.assertEqual(agents["planner"]["activity"], "Planning next action")
        self.assertEqual(agents["validator"]["status"], "idle")

    def test_empty_current_turn_never_replays_an_old_assistant_message(self):
        console, _runtime, hri, _calls = self.make_console()
        hri.hri_agent = HistoricalGraph()

        result = console.run_turn("Current question")

        self.assertTrue(result["ok"])
        self.assertNotIn("Earlier answer", result["assistant_text"])
        self.assertNotIn(
            "Earlier answer", json.dumps(console.cached_snapshot()["trace"])
        )

    def test_exact_controls_and_operator_emergency_use_runtime_boundary(self):
        console, runtime, _hri, _calls = self.make_console()

        self.assertTrue(console.confirm_goal("goal-1", 2)["ok"])
        self.assertTrue(console.decline_goal("goal-1", 2)["ok"])
        self.assertTrue(console.resume()["ok"])
        self.assertTrue(console.replan("Use the left side.")["ok"])
        stopped = console.emergency_stop("Operator saw a hazard.")

        self.assertEqual(
            runtime.confirmations,
            [("goal-1", 2, True), ("goal-1", 2, False)],
        )
        self.assertEqual(runtime.resume_count, 1)
        self.assertEqual(runtime.replans, ["Use the left side."])
        self.assertEqual(
            runtime.emergency.calls,
            [("Operator saw a hazard.", "publication-1")],
        )
        self.assertTrue(stopped["ok"])
        self.assertTrue(stopped["payload"]["accepted"])

    def test_tick_drains_notifications_into_bounded_non_consuming_ring(self):
        console, runtime, _hri, _calls = self.make_console(event_capacity=2)
        console.start()
        runtime.notifications.put("one")
        runtime.notifications.put({"kind": "REPORT", "summary": "two"})
        runtime.notifications.put("three")

        snapshot = console.tick()

        self.assertEqual(runtime.timeout_checks, 1)
        self.assertEqual([item["message"] for item in snapshot["events"]], ["two", "three"])
        cursor = snapshot["event_cursor"]
        self.assertEqual(console.events(after=cursor), [])
        self.assertEqual(console.events(), snapshot["events"])

    def test_tick_remains_available_during_a_blocking_model_turn(self):
        console, runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        result = {}

        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Wait")),
        )
        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))
        self.assertTrue(console.snapshot()["busy"])

        # This would block behind the model call if tick used the operation
        # lock.  It must remain an independent maintenance path.
        tick_result = console.tick()
        self.assertEqual(runtime.timeout_checks, 1)
        self.assertTrue(tick_result["busy"])

        graph.release.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["ok"])
        self.assertFalse(console.snapshot()["busy"])

    def test_emergency_stop_bypasses_a_blocking_model_turn(self):
        console, runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        result = {}

        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Wait")),
        )
        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))

        stopped = console.emergency_stop("Operator saw a hazard.")

        self.assertTrue(stopped["ok"])
        self.assertTrue(stopped["payload"]["accepted"])
        self.assertEqual(runtime.state, "EMERGENCY_STOPPED")
        self.assertTrue(worker.is_alive())

        graph.release.set()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())

    def test_emergency_stop_remains_available_while_close_drains_turn(self):
        console, runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        turn_result = {}
        close_result = {}
        stop_result = {}
        stop_finished = threading.Event()
        worker = threading.Thread(
            target=lambda: turn_result.update(console.run_turn("Wait")),
        )
        closer = threading.Thread(
            target=lambda: close_result.update(console.close()),
        )

        def stop_runtime() -> None:
            stop_result.update(console.emergency_stop("Operator saw a hazard."))
            stop_finished.set()

        stopper = threading.Thread(target=stop_runtime)
        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))
        closer.start()
        for _attempt in range(100):
            if console.closed:
                break
            threading.Event().wait(0.01)
        self.assertTrue(console.closed)
        stopper.start()

        self.assertTrue(stop_finished.wait(timeout=0.5))
        self.assertTrue(stop_result["ok"])
        self.assertEqual(
            runtime.emergency.calls,
            [("Operator saw a hazard.", "publication-1")],
        )
        self.assertTrue(closer.is_alive())

        graph.release.set()
        worker.join(timeout=1.0)
        closer.join(timeout=1.0)
        stopper.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertTrue(close_result["ok"])

    def test_failed_emergency_trigger_rolls_back_response_fence(self):
        console, runtime, _hri, _calls = self.make_console()
        console.start()
        original_trigger = runtime.emergency.trigger

        def fail_trigger(_reason, *, publication_id=None):
            del publication_id
            raise RuntimeError("stop transport failed")

        runtime.emergency.trigger = fail_trigger
        failed = console.emergency_stop("Operator saw a hazard.")
        runtime.emergency.trigger = original_trigger
        later_turn = console.run_turn("Can we still talk?")

        self.assertFalse(failed["ok"])
        self.assertEqual(failed["message"], "stop transport failed")
        self.assertFalse(runtime.emergency.latched)
        self.assertTrue(later_turn["ok"])
        self.assertEqual(later_turn["assistant_text"], "Visible answer")

    def test_runtime_latch_suppresses_reply_when_context_projection_fails(self):
        console, runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(console.run_turn("Wait")),
        )
        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))

        runtime.emergency.latched = True
        runtime.state = "EMERGENCY_STOPPED"

        def fail_context():
            raise RuntimeError("context projection failed")

        runtime.context_dict = fail_context
        graph.release.set()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(result["ok"])
        self.assertIn("emergency stop", result["message"].lower())
        self.assertTrue(result["context"]["emergency_latched"])
        snapshot = console.snapshot()
        self.assertTrue(snapshot["context"]["emergency_latched"])
        self.assertNotIn(
            "Finished",
            [item["content"] for item in snapshot["transcript"]],
        )

    def test_start_failure_is_user_facing_and_can_be_retried(self):
        runtime = FakeRuntime()
        hri = FakeHRI()
        attempts = 0

        def factory():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("camera unavailable")
            return runtime, hri

        console = AgentConsole(runtime_factory=factory)
        failed = console.start()
        recovered = console.start()

        self.assertFalse(failed["ok"])
        self.assertEqual(failed["message"], "camera unavailable")
        self.assertTrue(recovered["ok"])
        self.assertIsNone(console.snapshot()["startup_error"])
        self.assertFalse(console.snapshot()["busy"])

    def test_concurrent_start_is_visible_and_duplicate_attempts_do_not_block(self):
        runtime = FakeRuntime()
        hri = FakeHRI()
        factory_entered = threading.Event()
        release_factory = threading.Event()
        factory_calls = 0
        first_result = {}

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            factory_entered.set()
            if not release_factory.wait(timeout=2.0):
                raise TimeoutError("test did not release runtime startup")
            return runtime, hri

        console = AgentConsole(runtime_factory=factory)
        first = threading.Thread(
            target=lambda: first_result.update(console.start()),
        )
        first.start()
        self.assertTrue(factory_entered.wait(timeout=1.0))

        during_start = console.cached_snapshot()
        duplicate = console.start()
        duplicate_turn = console.run_turn("Do not queue this turn")
        duplicate_reset = console.reset_chat_session()

        self.assertTrue(during_start["busy"])
        self.assertEqual(during_start["busy_action"], "runtime startup")
        self.assertFalse(duplicate["ok"])
        self.assertEqual(duplicate["message"], "PrefMem is starting.")
        self.assertFalse(duplicate_turn["ok"])
        self.assertEqual(duplicate_turn["message"], "PrefMem is starting.")
        self.assertFalse(duplicate_reset["ok"])
        self.assertEqual(duplicate_reset["message"], "PrefMem is starting.")
        self.assertEqual(factory_calls, 1)
        self.assertTrue(first.is_alive())

        release_factory.set()
        first.join(timeout=1.0)
        self.assertFalse(first.is_alive())
        self.assertTrue(first_result["ok"])
        self.assertTrue(console.started)
        self.assertFalse(console.cached_snapshot()["busy"])

    def test_health_probes_are_parallel_and_report_degraded_camera(self):
        barrier = threading.Barrier(3)

        def fetch(url, timeout):
            self.assertEqual(timeout, 1.5)
            barrier.wait(timeout=1.0)
            if url.endswith("camera"):
                return {"status": "error", "error": "no frame"}
            return {"data": [{"id": url.rsplit("/", 1)[-1]}]}

        console = AgentConsole(
            runtime_factory=lambda: (FakeRuntime(), FakeHRI()),
            health_endpoints={
                "camera": "http://local/camera",
                "gemma": "http://local/gemma",
                "embedding": "http://local/embedding",
            },
            health_timeout=1.5,
            health_fetcher=fetch,
        )

        result = console.probe_health()

        self.assertEqual(set(result), {"camera", "gemma", "embedding"})
        self.assertEqual(result["camera"]["status"], "degraded")
        self.assertFalse(result["camera"]["ok"])
        self.assertTrue(result["gemma"]["ok"])
        self.assertEqual(console.snapshot()["health"], result)

    def test_health_probe_rejects_malformed_success_payload(self):
        console = AgentConsole(
            runtime_factory=lambda: (FakeRuntime(), FakeHRI()),
            health_endpoints={"gemma": "http://local/models"},
            health_fetcher=lambda _url, _timeout: {"unexpected": True},
        )

        result = console.probe_health()["gemma"]

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "degraded")
        self.assertIn("model list", result["error"])

    def test_close_is_idempotent_and_prevents_future_work(self):
        console, runtime, _hri, _calls = self.make_console()
        console.start()

        first = console.close()
        second = console.close()
        after = console.run_turn("Do something")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(runtime.close_count, 1)
        self.assertFalse(after["ok"])
        self.assertTrue(console.snapshot()["closed"])

    def test_close_waits_for_an_inflight_turn_and_turn_does_not_report_success(self):
        console, runtime, hri, _calls = self.make_console()
        graph = BlockingGraph()
        hri.hri_agent = graph
        turn_result = {}
        close_result = {}
        worker = threading.Thread(
            target=lambda: turn_result.update(console.run_turn("Wait")),
        )
        closer = threading.Thread(
            target=lambda: close_result.update(console.close()),
        )

        worker.start()
        self.assertTrue(graph.entered.wait(timeout=1.0))
        closer.start()
        for _attempt in range(100):
            if console.closed:
                break
            threading.Event().wait(0.01)
        self.assertTrue(console.snapshot()["closed"])
        self.assertTrue(closer.is_alive())

        graph.release.set()
        worker.join(timeout=1.0)
        closer.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertFalse(turn_result["ok"])
        self.assertTrue(close_result["ok"])
        self.assertEqual(runtime.close_count, 1)

    def test_repeated_close_preserves_a_shutdown_failure(self):
        console, runtime, _hri, _calls = self.make_console()
        console.start()

        def fail_close():
            runtime.close_count += 1
            raise RuntimeError("monitor join failed")

        runtime.close = fail_close
        first = console.close()
        second = console.close()

        self.assertFalse(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["message"], "monitor join failed")
        self.assertEqual(runtime.close_count, 1)


if __name__ == "__main__":
    unittest.main()
