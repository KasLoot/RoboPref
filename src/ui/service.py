"""Thread-safe application service for the RoboPref operator console.

The NiceGUI page is intentionally a thin projection of this class.  Keeping
the runtime boundary here makes the long, synchronous model calls easy to run
off the web event loop and leaves timeout polling and notification draining
available while a turn is in progress.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from prefmem.agents.metrics import message_text, reasoning_texts


DEFAULT_HEALTH_ENDPOINTS = {
    "camera": "http://127.0.0.1:1234/healthz",
    "gemma": "http://localhost:8000/v1/models",
    "embedding": "http://localhost:8080/v1/models",
}
DEFAULT_GUI_THREAD_ID = "prefmem-gui"
TRACE_CONTENT_LIMIT = 32_000


def _default_runtime_factory() -> tuple[Any, Any]:
    """Build the documented vLLM runtime without consuming process arguments."""

    from prefmem.cli import parse_args
    from prefmem.runtime import build_runtime

    return build_runtime(parse_args([]))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_copy(value: Any) -> Any:
    """Return a detached JSON-compatible value without trusting model types."""

    def fallback(item: Any) -> Any:
        for method_name in ("model_dump", "to_dict"):
            method = getattr(item, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    pass
        return str(item)

    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, default=fallback)
        )
    except Exception:
        return str(value)


def _default_health_fetcher(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "RoboPref-Operator-Console/2.1",
        },
    )
    try:
        # Local health checks must not leak into an ambient HTTP(S) proxy.
        with build_opener(ProxyHandler({})).open(
            request,
            timeout=timeout,
        ) as response:
            body = response.read(1_048_577)
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise RuntimeError(str(reason)) from error

    if len(body) > 1_048_576:
        raise RuntimeError("health response exceeds 1 MiB")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("health response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("health response must be a JSON object")
    return payload


class AgentConsole:
    """Own one lazily constructed PrefMem runtime and its GUI-safe state.

    ``runtime_factory`` returns ``(runtime, hri_agent)``.  Model/control
    operations are serialized through a dedicated lock.  ``tick`` deliberately
    does not use that lock, so monitor timeouts and notifications remain live
    during a long model request.
    """

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], tuple[Any, Any]] | None = None,
        thread_id: str | None = None,
        event_capacity: int = 200,
        transcript_capacity: int = 200,
        trace_capacity: int = 800,
        health_endpoints: Mapping[str, str] | None = None,
        health_timeout: float = 2.0,
        health_fetcher: Callable[[str, float], dict[str, Any]] | None = None,
    ) -> None:
        if runtime_factory is not None and not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        if event_capacity < 1:
            raise ValueError("event_capacity must be at least one")
        if transcript_capacity < 1:
            raise ValueError("transcript_capacity must be at least one")
        if trace_capacity < 1:
            raise ValueError("trace_capacity must be at least one")
        if health_timeout <= 0:
            raise ValueError("health_timeout must be greater than zero")

        if thread_id is not None and (
            not isinstance(thread_id, str) or not thread_id.strip()
        ):
            raise ValueError("thread_id must be a non-empty string")
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self.thread_id = (thread_id or DEFAULT_GUI_THREAD_ID).strip()
        self._health_endpoints = dict(
            DEFAULT_HEALTH_ENDPOINTS
            if health_endpoints is None
            else health_endpoints
        )
        self._health_timeout = float(health_timeout)
        self._health_fetcher = health_fetcher or _default_health_fetcher
        self._thread_id_prefix = self.thread_id

        self._state_lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._health_probe_lock = threading.Lock()
        self._emergency_lock = threading.Lock()
        self._runtime: Any | None = None
        self._hri: Any | None = None
        self._closed = False
        self._close_complete = False
        self._close_error: str | None = None
        self._emergency_requested = False
        self._busy = False
        self._busy_action: str | None = None
        self._startup_error: str | None = None
        self._last_error: str | None = None
        self._last_metrics: dict[str, Any] | None = None
        self._last_context: dict[str, Any] = self._offline_context()
        self._health: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=event_capacity)
        self._event_cursor = 0
        self._transcript: deque[dict[str, Any]] = deque(
            maxlen=transcript_capacity
        )
        self._message_cursor = 0
        self._trace: deque[dict[str, Any]] = deque(maxlen=trace_capacity)
        self._trace_cursor = 0
        self._turn_cursor = 0
        self._current_turn: dict[str, Any] | None = None
        self._agent_status: dict[str, dict[str, Any]] = {
            name: {"status": "offline", "activity": None}
            for name in ("hri", "planner", "memory", "monitor", "validator")
        }

    @staticmethod
    def _offline_context() -> dict[str, Any]:
        return {
            "state": "OFFLINE",
            "goal": None,
            "cycle_id": 0,
            "current_task": None,
            "execution_history": [],
            "attention_kind": None,
            "attention_reason": None,
            "latest_observation": None,
            "validation": None,
            "emergency_latched": False,
        }

    @staticmethod
    def _result(
        ok: bool,
        message: str,
        *,
        payload: Any = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "ok": bool(ok),
            "message": str(message),
            "payload": payload,
            **extra,
        }

    @property
    def runtime(self) -> Any | None:
        with self._state_lock:
            return self._runtime

    @property
    def hri(self) -> Any | None:
        with self._state_lock:
            return self._hri

    @property
    def started(self) -> bool:
        with self._state_lock:
            return self._runtime is not None and self._hri is not None

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def start(self) -> dict[str, Any]:
        """Start once; concurrent callers return busy and failures stay retriable."""

        if not self._start_lock.acquire(blocking=False):
            return self._busy_result()
        try:
            with self._state_lock:
                if self._closed:
                    return self._result(False, "The operator console is closed.")
                if self._runtime is not None and self._hri is not None:
                    context = self._safe_context()
                    return self._result(
                        True,
                        "PrefMem is already running.",
                        payload=context,
                        context=context,
                    )
                self._busy = True
                self._busy_action = "runtime startup"
                self._startup_error = None
                self._last_error = None
                for agent in self._agent_status.values():
                    agent["status"] = "starting"
                    agent["activity"] = "Starting PrefMem runtime"
            runtime: Any | None = None
            try:
                built = self._runtime_factory()
                if not isinstance(built, tuple) or len(built) != 2:
                    raise TypeError(
                        "runtime_factory must return (runtime, hri_agent)"
                    )
                runtime, hri = built
                self._validate_dependencies(runtime, hri)
                context = self._runtime_context(runtime)
            except Exception as error:
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception:
                        pass
                message = self._error_text(error)
                with self._state_lock:
                    self._startup_error = message
                    self._last_error = message
                    for agent in self._agent_status.values():
                        agent["status"] = "error"
                        agent["activity"] = message
                return self._result(False, message)

            with self._state_lock:
                if self._closed:
                    try:
                        runtime.close()
                    except Exception:
                        pass
                    return self._result(False, "The operator console is closed.")
                self._runtime = runtime
                self._hri = hri
                self._startup_error = None
                self._last_error = None
                self._last_context = context
                for agent in self._agent_status.values():
                    agent["status"] = "idle"
                    agent["activity"] = None
                self._append_event_locked(
                    "SYSTEM",
                    "PrefMem runtime started.",
                    level="success",
                )
            return self._result(
                True,
                "PrefMem runtime started.",
                payload=_json_copy(context),
                context=_json_copy(context),
            )
        finally:
            with self._state_lock:
                if self._busy_action == "runtime startup":
                    self._busy = False
                    self._busy_action = None
            self._start_lock.release()

    def run_turn(self, user_text: str) -> dict[str, Any]:
        """Compatibility wrapper which consumes :meth:`stream_turn`."""

        result: dict[str, Any] | None = None
        for event in self.stream_turn(user_text):
            if event.get("kind") == "turn_complete":
                payload = event.get("payload")
                if isinstance(payload, dict) and isinstance(
                    payload.get("result"), dict
                ):
                    result = payload["result"]
        if result is None:
            return self._result(
                False,
                "The HRI turn ended without a terminal result.",
                assistant_text="",
                context=self.context(),
            )
        return _json_copy(result)

    def stream_turn(self, user_text: str) -> Iterator[dict[str, Any]]:
        """Yield JSON-safe, current-turn graph events and one terminal result."""

        turn_id = self._next_turn_id()
        if not isinstance(user_text, str):
            result = self._result(False, "Message must be text.", assistant_text="")
            yield self._terminal_trace(turn_id, result, status="error")
            return
        text = user_text.strip()
        if not text:
            result = self._result(
                False,
                "Enter a message before sending.",
                assistant_text="",
            )
            yield self._terminal_trace(turn_id, result, status="error")
            return

        started = self.start()
        if not started["ok"]:
            result = {
                **started,
                "assistant_text": "",
                "context": self.cached_snapshot()["context"],
            }
            yield self._terminal_trace(turn_id, result, status="error")
            return
        if not self._begin_operation("HRI turn"):
            result = self._busy_result(assistant_text="")
            yield self._terminal_trace(turn_id, result, status="error")
            return

        turn_started = perf_counter()
        result: dict[str, Any]
        status = "error"
        archive_content: str | None = None
        archive_error = False
        self._begin_active_turn(turn_id)
        self._set_agent_status("hri", "working", "Starting HRI turn")
        self._append_transcript("user", text, turn_id=turn_id)
        try:
            yield self._append_trace_event(
                turn_id,
                "turn_start",
                node="HRI Agent",
                content=text,
            )
            with self._state_lock:
                runtime = self._runtime
                hri = self._hri
                thread_id = self.thread_id
            if runtime is None or hri is None:
                raise RuntimeError("PrefMem did not finish starting")

            frame = runtime.frame_source()
            image_block = getattr(frame, "image_block", None)
            if not isinstance(image_block, dict):
                raise TypeError("camera frame did not contain a model image block")

            completed_messages: list[Any] = []
            for spec in self._graph_turn_events(
                hri.hri_agent,
                text=text,
                image_block=image_block,
                thread_id=thread_id,
                runtime=runtime,
                completed_messages=completed_messages,
            ):
                event = self._append_trace_event(turn_id, **spec)
                self._apply_stream_event(event)
                yield event

            self._ensure_not_closed("The operator console closed during the turn.")
            assistant_text = self._visible_assistant_text(
                {"messages": completed_messages}
            )
            if not assistant_text:
                assistant_text = (
                    "The agent completed the turn without a visible text response."
                )
            context = self._safe_context()
            runtime_latched = self._runtime_emergency_latched(runtime)
            with self._state_lock:
                stop_requested = self._emergency_requested
            if stop_requested or runtime_latched or context.get("emergency_latched"):
                self._drain_notifications(runtime, hri, context)
                message = (
                    "A software emergency stop was latched during the turn; "
                    "the in-flight model response was discarded."
                )
                archive_content = message
                archive_error = True
                result = self._result(
                    False,
                    message,
                    assistant_text="",
                    context=_json_copy(context),
                )
                status = "emergency"
            else:
                archive_content = assistant_text
                self._drain_notifications(runtime, hri, context)
                elapsed = perf_counter() - turn_started
                metrics = self._metrics_summary(
                    getattr(hri, "metrics", None),
                    turn_seconds=elapsed,
                )
                payload = {
                    "context": _json_copy(context),
                    "elapsed_seconds": elapsed,
                    "metrics": _json_copy(metrics),
                }
                with self._state_lock:
                    self._last_error = None
                    self._last_metrics = metrics
                result = self._result(
                    True,
                    assistant_text,
                    payload=payload,
                    assistant_text=assistant_text,
                    context=_json_copy(context),
                )
                status = "complete"
        except Exception as error:
            message = self._error_text(error)
            self._record_error("HRI turn failed", message)
            archive_content = f"I could not complete that turn: {message}"
            archive_error = True
            result = self._result(
                False,
                message,
                assistant_text="",
                context=self._safe_context(),
            )
        finally:
            self._finalize_active_turn(
                turn_id,
                status=status,
                result=result if "result" in locals() else None,
                content=archive_content,
                error=archive_error,
            )
            if status == "emergency":
                for name in ("hri", "planner", "memory"):
                    self._set_agent_status(name, "stopped", None)
            else:
                self._set_agent_status("hri", "idle", None)
                self._set_agent_status(
                    "planner", "idle", None, only_if_running=True
                )
                self._set_agent_status(
                    "memory", "idle", None, only_if_running=True
                )
                self._set_agent_status(
                    "validator", "idle", None, only_if_running=True
                )
            # Create the terminal event before releasing the operation lock.
            # A reset can otherwise clear the old trace after lock release and
            # then receive this stale event in its new checkpoint thread.
            terminal_event = self._terminal_trace(turn_id, result, status=status)
            self._end_operation()

        yield terminal_event

    def reset_chat_session(self) -> dict[str, Any]:
        """Start a fresh HRI checkpoint thread without resetting the runtime."""

        # Match close()'s start-lock -> operation-lock order.  This prevents a
        # reset from racing the point where a newly built runtime/HRI pair is
        # installed, without starting a dormant runtime just to clear chat.
        if not self._start_lock.acquire(blocking=False):
            return self._busy_result()
        operation_started = False
        cleanup_error: str | None = None
        try:
            if not self._begin_operation("chat reset"):
                return self._busy_result()
            operation_started = True
            with self._state_lock:
                old_thread_id = self.thread_id
                self.thread_id = f"{self._thread_id_prefix}-{uuid.uuid4().hex}"
                new_thread_id = self.thread_id
                hri = self._hri
                self._transcript.clear()
                self._trace.clear()
                self._current_turn = None
                self._last_metrics = None

            # This list is conversational de-duplication state, not the durable
            # preference store owned by the Memory Agent.
            completed_mutations = getattr(
                hri, "completed_memory_mutations", None
            )
            if isinstance(completed_mutations, list):
                completed_mutations.clear()

            graph = getattr(hri, "hri_agent", None)
            checkpointer = getattr(graph, "checkpointer", None)
            delete_thread = getattr(checkpointer, "delete_thread", None)
            if callable(delete_thread):
                try:
                    delete_thread(old_thread_id)
                except Exception as error:
                    cleanup_error = self._error_text(error)

            with self._state_lock:
                self._append_event_locked(
                    "SYSTEM",
                    "Started a fresh HRI chat session.",
                    level="info" if cleanup_error is None else "warning",
                    payload={
                        "old_thread_id": old_thread_id,
                        "new_thread_id": new_thread_id,
                        "checkpoint_cleanup_error": cleanup_error,
                    },
                )
            context = self.cached_snapshot()["context"]
            message = "Started a fresh HRI chat session."
            if cleanup_error is not None:
                message += " The unused prior checkpoint could not be deleted."
            return self._result(
                True,
                message,
                payload={
                    "old_thread_id": old_thread_id,
                    "new_thread_id": new_thread_id,
                    "checkpoint_cleanup_error": cleanup_error,
                },
                context=context,
            )
        finally:
            if operation_started:
                self._end_operation()
            self._start_lock.release()

    def confirm_goal(self, goal_id: str, revision: int) -> dict[str, Any]:
        return self._control(
            "goal confirmation",
            lambda runtime: runtime.confirm_goal(
                goal_id=goal_id,
                revision=revision,
                confirmed=True,
            ),
            success_message="Goal confirmed; planning has started.",
        )

    def decline_goal(self, goal_id: str, revision: int) -> dict[str, Any]:
        return self._control(
            "goal decline",
            lambda runtime: runtime.confirm_goal(
                goal_id=goal_id,
                revision=revision,
                confirmed=False,
            ),
            success_message="Goal proposal declined.",
        )

    def resume(self) -> dict[str, Any]:
        return self._control(
            "resume",
            lambda runtime: runtime.resume_current_task(),
            success_message="Current task resumed.",
        )

    def resume_current_task(self) -> dict[str, Any]:
        """Compatibility name mirroring the runtime boundary."""

        return self.resume()

    def replan(self, guidance: str | None = None) -> dict[str, Any]:
        clean_guidance = guidance.strip() if isinstance(guidance, str) else None
        return self._control(
            "replan",
            lambda runtime: runtime.request_replan(clean_guidance or None),
            success_message="A fresh planning cycle was requested.",
        )

    def request_replan(self, guidance: str | None = None) -> dict[str, Any]:
        """Compatibility name mirroring the runtime boundary."""

        return self.replan(guidance)

    def emergency_stop(self, reason: str | None = None) -> dict[str, Any]:
        """Latch PrefMem through its coordinator (not a hardware-rated stop)."""

        with self._state_lock:
            runtime = self._runtime
            close_complete = self._close_complete
        if runtime is None:
            started = self.start()
            if not started["ok"]:
                return started
            with self._state_lock:
                runtime = self._runtime
        elif close_complete:
            return self._result(False, "The operator console is closed.")
        # EmergencyStopCoordinator is itself thread-safe.  Deliberately bypass
        # the start/model/control operation locks so an operator can latch the
        # runtime while a 300-second HRI request or graceful close is in flight.
        try:
            if runtime is None:
                raise RuntimeError("PrefMem did not finish starting")
            # Stop authority must not depend on a healthy context projection.
            context = self._safe_context()
            task = context.get("current_task")
            publication_id = (
                task.get("publication_id") if isinstance(task, dict) else None
            )
            if reason is not None and (
                not isinstance(reason, str) or not reason.strip()
            ):
                return self._result(False, "An emergency-stop reason is required.")
            clean_reason = (
                reason.strip()
                if isinstance(reason, str)
                else "Operator requested a software emergency stop."
            )
            with self._emergency_lock:
                with self._state_lock:
                    # Fence any response already being generated before invoking
                    # the runtime's independent, thread-safe stop coordinator.
                    previous_fence = self._emergency_requested
                    self._emergency_requested = True
                try:
                    accepted = bool(
                        runtime.emergency.trigger(
                            clean_reason,
                            publication_id=publication_id,
                        )
                    )
                except BaseException:
                    latched = self._runtime_emergency_latched(runtime)
                    with self._state_lock:
                        self._emergency_requested = previous_fence or latched
                    raise
                latched = self._runtime_emergency_latched(runtime)
                with self._state_lock:
                    self._emergency_requested = (
                        previous_fence or accepted or latched
                    )
                    if accepted or latched:
                        for agent in self._agent_status.values():
                            agent["status"] = "stopped"
                            agent["activity"] = None
            context = self._safe_context()
            self._drain_notifications(runtime, self._hri, context)
            message = (
                "Software emergency stop latched."
                if accepted
                else "Software emergency stop was already latched."
            )
            with self._state_lock:
                self._append_event_locked(
                    "EMERGENCY_STOP",
                    message,
                    level="critical",
                    payload={
                        "reason": clean_reason,
                        "publication_id": publication_id,
                    },
                )
            return self._result(
                True,
                message,
                payload={"accepted": accepted, "context": _json_copy(context)},
                context=_json_copy(context),
            )
        except Exception as error:
            message = self._error_text(error)
            self._record_error("Software stop failed", message)
            return self._result(False, message)

    def _control(
        self,
        action: str,
        callback: Callable[[Any], Any],
        *,
        success_message: str,
    ) -> dict[str, Any]:
        started = self.start()
        if not started["ok"]:
            return started
        if not self._begin_operation(action):
            return self._busy_result()
        active_agents = self._control_agents(action)
        for name in active_agents:
            self._set_agent_status(name, "working", action.title())
        try:
            with self._state_lock:
                runtime = self._runtime
                hri = self._hri
            if runtime is None:
                raise RuntimeError("PrefMem did not finish starting")
            value = callback(runtime)
            self._ensure_not_closed(
                f"The operator console closed during {action}."
            )
            context = (
                _json_copy(value)
                if isinstance(value, dict)
                else self._runtime_context(runtime)
            )
            with self._state_lock:
                self._last_context = context
                self._last_error = None
                self._append_event_locked(
                    "OPERATOR_ACTION",
                    success_message,
                    level="success",
                    payload={"action": action},
                )
            self._drain_notifications(runtime, hri, context)
            return self._result(
                True,
                success_message,
                payload=_json_copy(context),
                context=_json_copy(context),
            )
        except Exception as error:
            message = self._error_text(error)
            self._record_error(f"{action.title()} failed", message)
            return self._result(False, message)
        finally:
            for name in active_agents:
                self._set_agent_status(name, "idle", None, only_if_running=True)
            self._end_operation()

    def tick(self) -> dict[str, Any]:
        """Run cheap maintenance without waiting for the operation lock."""

        if not self._maintenance_lock.acquire(blocking=False):
            return self.cached_snapshot()
        with self._state_lock:
            runtime = self._runtime
            hri = self._hri
            closed = self._closed
        try:
            if runtime is not None and not closed:
                try:
                    runtime.check_timeout()
                except Exception as error:
                    self._record_error("Timeout check failed", self._error_text(error))
                context = self._safe_context()
                self._drain_notifications(runtime, hri, context)
            return self.cached_snapshot()
        finally:
            self._maintenance_lock.release()

    def drain_notifications(self) -> dict[str, Any]:
        """Drain the runtime queue once into the shared, non-consuming ring."""

        with self._maintenance_lock:
            with self._state_lock:
                runtime = self._runtime
                hri = self._hri
            if runtime is not None:
                context = self._safe_context()
                self._drain_notifications(runtime, hri, context)
        return self.cached_snapshot()

    def probe_health(self) -> dict[str, dict[str, Any]]:
        """Probe camera and both model servers concurrently."""

        if not self._health_probe_lock.acquire(blocking=False):
            with self._state_lock:
                return _json_copy(self._health)
        endpoints = list(self._health_endpoints.items())
        try:
            if not endpoints:
                with self._state_lock:
                    self._health = {}
                return {}

            results: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(
                max_workers=len(endpoints),
                thread_name_prefix="robopref-health",
            ) as pool:
                pending = {
                    pool.submit(self._probe_one, name, url): name
                    for name, url in endpoints
                }
                for future in as_completed(pending):
                    name = pending[future]
                    try:
                        results[name] = future.result()
                    except Exception as error:  # defensive: _probe_one is total
                        results[name] = self._degraded_health(
                            self._health_endpoints[name],
                            self._error_text(error),
                        )

            ordered = {name: results[name] for name, _url in endpoints}
            with self._state_lock:
                self._health = _json_copy(ordered)
            return ordered
        finally:
            self._health_probe_lock.release()

    def _probe_one(self, name: str, url: str) -> dict[str, Any]:
        started_at = perf_counter()
        checked_at = _utc_now()
        try:
            payload = self._health_fetcher(url, self._health_timeout)
            if not isinstance(payload, dict):
                raise RuntimeError("health response must be a JSON object")
            if name == "camera":
                status = str(payload.get("status", "")).strip().lower()
                if status != "ok":
                    error = payload.get("error") or f"camera status is {status or 'missing'}"
                    raise RuntimeError(str(error))
                detail = {
                    key: payload.get(key)
                    for key in ("camera", "frames_captured")
                    if key in payload
                }
            else:
                models = payload.get("data")
                if not isinstance(models, list) or not models:
                    raise RuntimeError("model list is missing or empty")
                identifiers = [
                    item.get("id")
                    for item in models
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                if not identifiers:
                    raise RuntimeError("model list contains no model identifiers")
                detail = {"models": identifiers}
            return {
                "ok": True,
                "status": "online",
                "url": url,
                "checked_at": checked_at,
                "latency_ms": round((perf_counter() - started_at) * 1000, 1),
                "detail": detail,
                "payload": _json_copy(payload),
                "error": None,
            }
        except Exception as error:
            return self._degraded_health(
                url,
                self._error_text(error),
                checked_at=checked_at,
                latency_ms=round((perf_counter() - started_at) * 1000, 1),
            )

    @staticmethod
    def _degraded_health(
        url: str,
        error: str,
        *,
        checked_at: str | None = None,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "degraded",
            "url": url,
            "checked_at": checked_at or _utc_now(),
            "latency_ms": latency_ms,
            "detail": {},
            "payload": None,
            "error": error,
        }

    def events(self, *, after: int | None = None) -> list[dict[str, Any]]:
        with self._state_lock:
            values = list(self._events)
        if after is None:
            return _json_copy(values)
        if isinstance(after, bool) or not isinstance(after, int):
            raise TypeError("after must be a non-negative integer cursor")
        if after < 0:
            raise ValueError("after must be a non-negative integer cursor")
        return _json_copy(
            [item for item in values if int(item["cursor"]) > after]
        )

    def snapshot(self) -> dict[str, Any]:
        context = self._safe_context()
        return self._snapshot_from_context(context)

    def cached_snapshot(self) -> dict[str, Any]:
        """Return the latest projection without polling the PrefMem runtime."""

        with self._state_lock:
            context = _json_copy(self._last_context)
        return self._snapshot_from_context(context)

    def _snapshot_from_context(
        self,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._state_lock:
            started = self._runtime is not None and self._hri is not None
            ready = started and not self._closed
            services = self._service_status(self._runtime)
            return {
                "ready": ready,
                "started": started,
                "closed": self._closed,
                "busy": self._busy,
                "busy_action": self._busy_action,
                "busy_operation": self._busy_action,
                "startup_error": self._startup_error,
                "last_error": self._last_error,
                "thread_id": self.thread_id,
                "runtime": _json_copy(context),
                "context": _json_copy(context),
                "services": services,
                "agents": self._agents_snapshot(
                    started=started,
                    services=services,
                    context=context,
                ),
                "metrics": _json_copy(self._last_metrics),
                "health": _json_copy(self._health),
                "events": _json_copy(list(self._events)),
                "event_cursor": self._event_cursor,
                "transcript": _json_copy(list(self._transcript)),
                "message_cursor": self._message_cursor,
                "trace": _json_copy(list(self._trace)),
                "trace_cursor": self._trace_cursor,
                "active_turn": _json_copy(self._current_turn),
            }

    def context(self) -> dict[str, Any]:
        """Return authoritative context without starting a dormant runtime."""

        return self._safe_context()

    def close(self) -> dict[str, Any]:
        """Close the runtime exactly once and reject all subsequent work."""

        with self._start_lock:
            with self._state_lock:
                if self._closed:
                    if self._close_error is not None:
                        return self._result(False, self._close_error)
                    return self._result(True, "Operator console is already closed.")
                self._closed = True
                runtime = self._runtime
            # Mark closed before waiting so no new operation or maintenance
            # cycle can enter.  Then let any in-flight runtime call finish
            # before shutting down Monitor/Validator and clearing the display.
            with self._operation_lock, self._maintenance_lock:
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception as error:
                        message = self._error_text(error)
                        with self._state_lock:
                            self._close_error = message
                        self._record_error("Runtime shutdown failed", message)
                        return self._result(False, message)
            with self._state_lock:
                self._close_complete = True
                for agent in self._agent_status.values():
                    agent["status"] = "stopped"
                    agent["activity"] = None
                self._append_event_locked(
                    "SYSTEM",
                    "Operator console closed.",
                    level="info",
                )
            return self._result(True, "Operator console closed.")

    def _begin_operation(self, action: str) -> bool:
        with self._state_lock:
            if self._closed:
                return False
        if not self._operation_lock.acquire(blocking=False):
            return False
        with self._state_lock:
            if self._closed:
                self._operation_lock.release()
                return False
            self._busy = True
            self._busy_action = action
        return True

    def _ensure_not_closed(self, message: str) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError(message)

    @staticmethod
    def _runtime_emergency_latched(runtime: Any) -> bool:
        try:
            emergency = getattr(runtime, "emergency", None)
            return bool(getattr(emergency, "latched", False))
        except Exception:
            return False

    def _end_operation(self) -> None:
        with self._state_lock:
            self._busy = False
            self._busy_action = None
        self._operation_lock.release()

    def _busy_result(self, **extra: Any) -> dict[str, Any]:
        with self._state_lock:
            if self._closed:
                message = "The operator console is closed."
            elif self._busy_action == "runtime startup":
                message = "PrefMem is starting."
            elif self._busy_action:
                message = f"PrefMem is busy with {self._busy_action}."
            else:
                message = "PrefMem is busy."
        return self._result(False, message, **extra)

    def _runtime_context(self, runtime: Any) -> dict[str, Any]:
        value = runtime.context_dict()
        if not isinstance(value, dict):
            raise TypeError("runtime.context_dict() must return a dictionary")
        return _json_copy(value)

    @staticmethod
    def _validate_dependencies(runtime: Any, hri: Any) -> None:
        for method_name in (
            "context_dict",
            "frame_source",
            "confirm_goal",
            "resume_current_task",
            "request_replan",
            "check_timeout",
            "close",
        ):
            if not callable(getattr(runtime, method_name, None)):
                raise TypeError(f"runtime is missing callable {method_name}()")
        emergency = getattr(runtime, "emergency", None)
        if not callable(getattr(emergency, "trigger", None)):
            raise TypeError("runtime is missing emergency.trigger()")
        graph = getattr(hri, "hri_agent", None)
        if not callable(getattr(graph, "invoke", None)):
            raise TypeError("hri_agent is missing a compiled graph invoke()")

    @staticmethod
    def _service_status(runtime: Any | None) -> dict[str, dict[str, Any]]:
        statuses: dict[str, dict[str, Any]] = {}
        for name in ("monitor", "validator"):
            service = None if runtime is None else getattr(runtime, name, None)
            statuses[name] = {
                "running": bool(getattr(service, "running", False)),
                "active_publication_id": getattr(
                    service,
                    "active_publication_id",
                    None,
                ),
            }
        return statuses

    def _agents_snapshot(
        self,
        *,
        started: bool,
        services: Mapping[str, Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        emergency = bool(context.get("emergency_latched"))
        agents: dict[str, dict[str, Any]] = {}
        for name in ("hri", "planner", "memory"):
            value = self._agent_status[name]
            status = str(value.get("status") or "idle")
            activity = value.get("activity")
            if not started:
                status = "offline"
                activity = None
            elif self._closed or emergency:
                status = "stopped"
                activity = None
            elif (
                name == "planner"
                and str(context.get("state") or "").upper() == "PLANNING"
            ):
                status = "working"
                activity = "Planning next action"
            agents[name] = {
                "status": status,
                "activity": activity,
                "active_publication_id": None,
            }

        for name in ("monitor", "validator"):
            service = services.get(name, {})
            publication_id = service.get("active_publication_id")
            running = bool(service.get("running"))
            dynamic = self._agent_status[name]
            dynamic_running = dynamic.get("status") == "working"
            task = context.get("current_task")
            task = task if isinstance(task, Mapping) else {}
            task_publication_id = task.get("publication_id")
            phase = str(task.get("phase") or "").upper()
            runtime_state = str(context.get("state") or "").upper()
            if name == "validator" and runtime_state == "PLANNING":
                dynamic_running = False
            publication_matches = bool(
                publication_id
                and task_publication_id
                and publication_id == task_publication_id
            )
            phase_matches = (
                phase == "STEP" and runtime_state == "EXECUTING"
                if name == "monitor"
                else phase == "FINAL_VALIDATION"
                and runtime_state == "FINAL_VALIDATION"
            )
            background_active = running and publication_matches and phase_matches
            if not started:
                status = "offline"
            elif self._closed or emergency:
                status = "stopped"
            else:
                if dynamic_running:
                    status = "working"
                elif background_active:
                    status = "watching" if name == "monitor" else "working"
                else:
                    status = "idle"
            if dynamic_running and status == "working":
                activity = dynamic.get("activity")
            elif background_active and status in {"working", "watching"}:
                verb = "Monitoring" if name == "monitor" else "Validating"
                activity = (
                    f"{verb} {publication_id}"
                    if publication_id
                    else verb
                )
            else:
                activity = None
            agents[name] = {
                "status": status,
                "activity": activity,
                "active_publication_id": publication_id,
            }
        return _json_copy(agents)

    @staticmethod
    def _control_agents(action: str) -> tuple[str, ...]:
        if action == "goal confirmation":
            # Checklist compilation precedes planning. The context-derived
            # PLANNING projection above transfers working status to Planner.
            return ("validator",)
        if action == "replan":
            return ("planner",)
        return ()

    @staticmethod
    def _metrics_summary(
        metrics: Any,
        *,
        turn_seconds: float,
    ) -> dict[str, Any] | None:
        summary = getattr(metrics, "summary", None)
        if not callable(summary):
            return None
        try:
            value = summary(turn_seconds=turn_seconds)
        except Exception:
            return None
        return _json_copy(value) if isinstance(value, Mapping) else None

    def _refresh_context(self, runtime: Any) -> dict[str, Any]:
        context = self._runtime_context(runtime)
        with self._state_lock:
            self._last_context = context
        return _json_copy(context)

    def _safe_context(self) -> dict[str, Any]:
        with self._state_lock:
            runtime = self._runtime
            cached = _json_copy(self._last_context)
        if runtime is None:
            return cached
        try:
            context = self._refresh_context(runtime)
        except Exception:
            context = cached
        if self._runtime_emergency_latched(runtime) and not context.get(
            "emergency_latched"
        ):
            context = {
                **context,
                "state": "EMERGENCY_STOPPED",
                "emergency_latched": True,
            }
            with self._state_lock:
                self._last_context = context
        return _json_copy(context)

    def _graph_turn_events(
        self,
        graph: Any,
        *,
        text: str,
        image_block: dict[str, Any],
        thread_id: str,
        runtime: Any,
        completed_messages: list[Any],
    ) -> Iterator[dict[str, Any]]:
        graph_input = {"messages": [HumanMessage(content=text)]}
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 20,
        }
        context = {"current_frame": image_block}
        stream_method = getattr(graph, "stream", None)
        if not callable(stream_method):
            response = graph.invoke(graph_input, config=config, context=context)
            messages = self._current_turn_messages(response)
            completed_messages.extend(messages)
            yield from self._completed_message_events(messages)
            return

        iterator = stream_method(
            input=graph_input,
            config=config,
            context=context,
            stream_mode=["messages", "updates"],
            version="v2",
        )
        streamed = {
            "assistant": False,
            "thinking": False,
            "tool": False,
        }
        try:
            for part in iterator:
                with self._state_lock:
                    stop_requested = self._emergency_requested
                if stop_requested or self._runtime_emergency_latched(runtime):
                    break
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                data = part.get("data")
                if part_type == "messages":
                    if not isinstance(data, (list, tuple)) or len(data) != 2:
                        continue
                    chunk, metadata = data
                    if not isinstance(chunk, AIMessage) or not isinstance(
                        metadata, Mapping
                    ):
                        continue
                    if metadata.get("langgraph_node") != "HRI Agent":
                        continue
                    for reasoning in reasoning_texts(chunk):
                        if reasoning:
                            streamed["thinking"] = True
                            yield {
                                "kind": "thinking_delta",
                                "node": "HRI Agent",
                                "content": reasoning,
                            }
                    visible = message_text(chunk.content)
                    if visible:
                        streamed["assistant"] = True
                        yield {
                            "kind": "assistant_delta",
                            "node": "HRI Agent",
                            "content": visible,
                        }
                    for call in getattr(chunk, "tool_call_chunks", ()):
                        if not isinstance(call, Mapping):
                            continue
                        streamed["tool"] = True
                        name = call.get("name")
                        arguments = call.get("args") or ""
                        rendered_delta = (
                            f"{name} arguments={arguments}"
                            if name
                            else str(arguments)
                        )
                        yield {
                            "kind": "tool_call_delta",
                            "node": "HRI Agent",
                            "content": rendered_delta,
                            "payload": {
                                "index": call.get("index"),
                                "tool_call_id": call.get("id"),
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    continue

                if part_type != "updates" or not isinstance(data, Mapping):
                    continue
                for node_name, update in data.items():
                    if not isinstance(update, Mapping):
                        continue
                    messages = update.get("messages", ())
                    if not isinstance(messages, (list, tuple)):
                        continue
                    completed_messages.extend(messages)
                    if node_name == "HRI Agent":
                        for message in messages:
                            if not isinstance(message, AIMessage):
                                continue
                            if not streamed["thinking"]:
                                for reasoning in reasoning_texts(message):
                                    if reasoning:
                                        yield {
                                            "kind": "thinking_delta",
                                            "node": "HRI Agent",
                                            "content": reasoning,
                                            "payload": {"complete": True},
                                        }
                            visible = message_text(message.content)
                            if visible and not streamed["assistant"]:
                                yield {
                                    "kind": "assistant_delta",
                                    "node": "HRI Agent",
                                    "content": visible,
                                    "payload": {"complete": True},
                                }
                            for call in message.tool_calls:
                                yield {
                                    "kind": "tool_call",
                                    "node": "HRI Agent",
                                    "payload": self._tool_call_payload(call),
                                }
                        streamed = {
                            "assistant": False,
                            "thinking": False,
                            "tool": False,
                        }
                    elif node_name == "Tool Node":
                        for message in messages:
                            if not isinstance(message, ToolMessage):
                                continue
                            yield {
                                "kind": "tool_result",
                                "node": "Tool Node",
                                "content": self._message_content(message.content),
                                "payload": {
                                    "tool_call_id": message.tool_call_id,
                                    "name": message.name,
                                },
                            }
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _completed_message_events(
        self,
        messages: list[Any],
    ) -> Iterator[dict[str, Any]]:
        for message in messages:
            if isinstance(message, AIMessage):
                for reasoning in reasoning_texts(message):
                    if reasoning:
                        yield {
                            "kind": "thinking_delta",
                            "node": "HRI Agent",
                            "content": reasoning,
                            "payload": {"complete": True},
                        }
                visible = message_text(message.content)
                if visible:
                    yield {
                        "kind": "assistant_delta",
                        "node": "HRI Agent",
                        "content": visible,
                        "payload": {"complete": True},
                    }
                for call in message.tool_calls:
                    yield {
                        "kind": "tool_call",
                        "node": "HRI Agent",
                        "payload": self._tool_call_payload(call),
                    }
            elif isinstance(message, ToolMessage):
                yield {
                    "kind": "tool_result",
                    "node": "Tool Node",
                    "content": self._message_content(message.content),
                    "payload": {
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                    },
                }

    @staticmethod
    def _current_turn_messages(response: Any) -> list[Any]:
        messages = response.get("messages", ()) if isinstance(response, dict) else ()
        if not isinstance(messages, (list, tuple)):
            return []
        latest_human = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        return list(messages[latest_human + 1 :] if latest_human is not None else messages)

    @staticmethod
    def _tool_call_payload(call: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "index": call.get("index"),
            "tool_call_id": call.get("id"),
            "name": call.get("name"),
            "arguments": _json_copy(call.get("args", {})),
        }

    @staticmethod
    def _message_content(content: Any) -> str:
        rendered = message_text(content)
        if rendered:
            return rendered
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    @staticmethod
    def _visible_assistant_text(response: Any) -> str:
        messages = response.get("messages", []) if isinstance(response, dict) else []
        if not isinstance(messages, (list, tuple)):
            return ""
        latest_human = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        current_turn = (
            messages[latest_human + 1 :]
            if latest_human is not None
            else messages
        )
        for message in reversed(current_turn):
            if not isinstance(message, AIMessage):
                continue
            if message.tool_calls or message.invalid_tool_calls:
                continue
            text = message_text(message.content).strip()
            if text:
                return text
        return ""

    def _next_turn_id(self) -> str:
        with self._state_lock:
            self._turn_cursor += 1
            return f"turn-{self._turn_cursor}"

    def _append_trace_event(
        self,
        turn_id: str,
        kind: str,
        *,
        node: str | None = None,
        content: Any = None,
        payload: Any = None,
    ) -> dict[str, Any]:
        rendered = None if content is None else str(content)
        event_payload = _json_copy(payload if payload is not None else {})
        if rendered is not None and len(rendered) > TRACE_CONTENT_LIMIT:
            rendered = rendered[:TRACE_CONTENT_LIMIT]
            if isinstance(event_payload, dict):
                event_payload["truncated"] = True
        with self._state_lock:
            self._trace_cursor += 1
            event = {
                "sequence": self._trace_cursor,
                "turn_id": turn_id,
                "kind": kind,
                "node": node,
                "content": rendered,
                "payload": event_payload,
                "timestamp": _utc_now(),
            }
            self._trace.append(event)
        return _json_copy(event)

    def _terminal_trace(
        self,
        turn_id: str,
        result: Mapping[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        return self._append_trace_event(
            turn_id,
            "turn_complete",
            node="HRI Agent",
            content=result.get("message"),
            payload={"status": status, "result": _json_copy(result)},
        )

    def _begin_active_turn(self, turn_id: str) -> None:
        with self._state_lock:
            self._current_turn = {
                "turn_id": turn_id,
                "revision": 1,
                "status": "streaming",
                "assistant_text": "",
                "reasoning_text": "",
                "tools": [],
                "started_at": _utc_now(),
                "finished_at": None,
            }

    def _apply_stream_event(self, event: Mapping[str, Any]) -> None:
        kind = event.get("kind")
        content = str(event.get("content") or "")
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        agents_to_start: tuple[str, ...] = ()
        agents_to_finish: tuple[str, ...] = ()
        with self._state_lock:
            active = self._current_turn
            if active is None or active.get("turn_id") != event.get("turn_id"):
                return
            if kind == "assistant_delta" and content:
                active["assistant_text"] = self._bounded_text(
                    str(active.get("assistant_text") or "") + content
                )
            elif kind == "thinking_delta" and content:
                active["reasoning_text"] = self._bounded_text(
                    str(active.get("reasoning_text") or "") + content
                )
            elif kind in {"tool_call_delta", "tool_call"}:
                tool = self._active_tool(active, payload)
                if payload.get("name"):
                    tool["name"] = str(payload["name"])
                if kind == "tool_call_delta":
                    fragment = str(payload.get("arguments") or "")
                    tool["arguments"] = self._bounded_text(
                        str(tool.get("arguments") or "") + fragment
                    )
                else:
                    tool["arguments"] = self._bounded_value(
                        payload.get("arguments", {})
                    )
                    tool["status"] = "running"
                    agents_to_start = self._tool_agents(
                        tool.get("name"), tool.get("arguments")
                    )
            elif kind == "tool_result":
                tool = self._active_tool(active, payload)
                tool["result"] = self._bounded_text(content)
                tool["status"] = "complete"
                agents_to_finish = self._tool_agents(
                    tool.get("name"), tool.get("arguments")
                )
            active["revision"] = int(active.get("revision", 0)) + 1
        for agent_to_start in agents_to_start:
            self._set_agent_status(
                agent_to_start,
                "working",
                f"Running {payload.get('name') or 'tool'}",
            )
        for agent_to_finish in agents_to_finish:
            self._set_agent_status(agent_to_finish, "idle", None)
        if kind == "thinking_delta":
            self._set_agent_status("hri", "working", "Thinking")
        elif kind == "assistant_delta":
            self._set_agent_status("hri", "working", "Responding")

    def _active_tool(
        self,
        active: dict[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        tools = active.setdefault("tools", [])
        call_id = payload.get("tool_call_id")
        index = payload.get("index")
        name = payload.get("name")
        for tool in reversed(tools):
            if call_id and tool.get("tool_call_id") == call_id:
                return tool
            if call_id is None and index is not None and tool.get("index") == index:
                return tool
        tool = {
            "tool_call_id": call_id,
            "index": index,
            "name": name,
            "arguments": "",
            "result": None,
            "status": "streaming",
        }
        tools.append(tool)
        if len(tools) > 20:
            del tools[:-20]
        return tool

    def _finalize_active_turn(
        self,
        turn_id: str,
        *,
        status: str,
        result: Mapping[str, Any] | None,
        content: str | None,
        error: bool,
    ) -> None:
        """Atomically replace the live draft with its transcript record."""

        with self._state_lock:
            active = self._current_turn
            if active is not None and active.get("turn_id") == turn_id:
                active["status"] = status
                active["finished_at"] = _utc_now()
                if isinstance(result, Mapping) and isinstance(
                    result.get("assistant_text"), str
                ):
                    active["assistant_text"] = self._bounded_text(
                        result["assistant_text"]
                    )
                active["revision"] = int(active.get("revision", 0)) + 1
            if content is not None:
                self._append_transcript(
                    "assistant",
                    content,
                    error=error,
                    turn_id=turn_id,
                )
            if active is not None and active.get("turn_id") == turn_id:
                self._current_turn = None

    @staticmethod
    def _bounded_text(value: str) -> str:
        return value[:TRACE_CONTENT_LIMIT]

    @classmethod
    def _bounded_value(cls, value: Any) -> Any:
        copied = _json_copy(value)
        rendered = json.dumps(copied, ensure_ascii=False, default=str)
        if len(rendered) <= TRACE_CONTENT_LIMIT:
            return copied
        return {
            "truncated": True,
            "preview": rendered[:TRACE_CONTENT_LIMIT],
        }

    @staticmethod
    def _tool_agents(
        name: Any,
        arguments: Any = None,
    ) -> tuple[str, ...]:
        normalized = str(name or "")
        if normalized == "call_memory_agent":
            return ("memory",)
        if normalized == "request_goal_preview":
            return ("planner",)
        if normalized == "confirm_goal_execution":
            if isinstance(arguments, Mapping) and not arguments.get(
                "confirmed", True
            ):
                return ()
            return ("validator",)
        if normalized == "request_execution_replan":
            return ("planner",)
        return ()

    def _set_agent_status(
        self,
        name: str,
        status: str,
        activity: str | None,
        *,
        only_if_running: bool = False,
    ) -> None:
        with self._state_lock:
            current = self._agent_status.get(name)
            if current is None:
                return
            if only_if_running and current.get("status") != "working":
                return
            current["status"] = status
            current["activity"] = activity

    def _append_transcript(
        self,
        role: str,
        content: str,
        *,
        error: bool = False,
        turn_id: str | None = None,
    ) -> None:
        with self._state_lock:
            self._message_cursor += 1
            self._transcript.append(
                {
                    "id": self._message_cursor,
                    "role": role,
                    "content": content,
                    "error": error,
                    "turn_id": turn_id,
                    "timestamp": _utc_now(),
                }
            )

    def _drain_notifications(
        self,
        runtime: Any,
        hri: Any,
        context: Mapping[str, Any],
    ) -> None:
        notifications = getattr(runtime, "notifications", None)
        if notifications is None:
            return
        state = context.get("state")
        while True:
            try:
                value = notifications.get_nowait()
            except queue.Empty:
                return
            try:
                renderer = getattr(hri, "_render_runtime_notification", None)
                rendered = (
                    renderer(value, authoritative_state=state)
                    if callable(renderer)
                    else str(value)
                )
            except Exception:
                rendered = str(value)
            level = "critical" if "EMERGENCY" in str(rendered).upper() else "info"
            with self._state_lock:
                self._append_event_locked(
                    "RUNTIME_NOTIFICATION",
                    str(rendered),
                    level=level,
                    payload=_json_copy(value),
                )

    def _append_event_locked(
        self,
        kind: str,
        message: str,
        *,
        level: str,
        payload: Any = None,
    ) -> None:
        self._event_cursor += 1
        created_at = _utc_now()
        self._events.append(
            {
                "sequence": self._event_cursor,
                "cursor": self._event_cursor,
                "kind": kind,
                "level": level,
                "message": message,
                "payload": _json_copy(payload),
                "created_at": created_at,
                "timestamp": created_at,
            }
        )

    def _record_error(self, label: str, message: str) -> None:
        with self._state_lock:
            self._last_error = message
            self._append_event_locked(
                "ERROR",
                f"{label}: {message}",
                level="error",
            )

    @staticmethod
    def _error_text(error: BaseException) -> str:
        text = " ".join(str(error).split())
        return text or type(error).__name__


__all__ = [
    "AgentConsole",
    "DEFAULT_GUI_THREAD_ID",
    "DEFAULT_HEALTH_ENDPOINTS",
]
