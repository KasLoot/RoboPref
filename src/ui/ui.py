"""NiceGUI operator workspace for the RoboPref agent runtime."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nicegui import app, run, ui

from ui.service import AgentConsole
from ui.theme import APP_CSS


DEFAULT_GUI_HOST = "127.0.0.1"
DEFAULT_GUI_PORT = 8090
DEFAULT_CAMERA_BASE_URL = "http://127.0.0.1:1234"


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    camera_base_url: str = DEFAULT_CAMERA_BASE_URL
    memory_store_path: Path = Path("./memory_store")
    port: int = DEFAULT_GUI_PORT
    think: tuple[str, ...] = ("HRI",)
    open_browser: bool = False


STATE_META = {
    "OFFLINE": ("Offline", "power_settings_new", "offline"),
    "IDLE": ("Ready", "radio_button_checked", "ready"),
    "AWAITING_CONFIRMATION": ("Review goal", "fact_check", "attention"),
    "PLANNING": ("Planning", "route", "working"),
    "EXECUTING": ("Executing", "precision_manufacturing", "working"),
    "FINAL_VALIDATION": ("Validating", "verified", "working"),
    "NEEDS_ATTENTION": ("Needs attention", "report_problem", "attention"),
    "COMPLETE": ("Complete", "task_alt", "complete"),
    "EMERGENCY_STOPPED": ("Software stop", "emergency", "danger"),
}

AGENT_META = {
    "hri": ("HRI", "forum"),
    "planner": ("Planner", "route"),
    "memory": ("Memory", "database"),
    "validator": ("Validator", "verified"),
}


def _http_base_url(value: str) -> str:
    if not value or any(
        character.isspace()
        or ord(character) < 0x20
        or ord(character) == 0x7F
        or character in {'"', "'", "`", "\\"}
        for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must be a plain HTTP(S) origin without whitespace or quoting"
        )
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must contain a valid host and optional TCP port"
        ) from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise argparse.ArgumentTypeError(
            "must be an absolute HTTP(S) origin without credentials or a path"
        )

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if len(hostname) > 253:
            raise argparse.ArgumentTypeError("host name is too long")
        labels = hostname.split(".")
        valid_labels = all(
            label
            and len(label) <= 63
            and re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?",
                label,
            )
            for label in labels
        )
        if not valid_labels:
            raise argparse.ArgumentTypeError(
                "host name contains invalid characters"
            )
        canonical_host = hostname.lower()
    else:
        canonical_host = address.compressed
        if address.version == 6:
            canonical_host = f"[{canonical_host}]"

    port_suffix = f":{port}" if port is not None else ""
    return f"{parsed.scheme.lower()}://{canonical_host}{port_suffix}"


def _port(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= result <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ui",
        description="Run the local RoboPref NiceGUI operator workspace.",
    )
    parser.add_argument(
        "--camera-base-url",
        type=_http_base_url,
        default=DEFAULT_CAMERA_BASE_URL,
        help="camera stream origin (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-store-path",
        type=Path,
        default=Path("./memory_store"),
        metavar="PATH",
        help="persistent preference-memory directory",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_GUI_PORT,
        help="loopback GUI port (default: %(default)s)",
    )
    parser.add_argument(
        "--think",
        nargs="*",
        default=["HRI"],
        metavar="AGENT",
        help="agents allowed to use model thinking (default: HRI)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the workspace in the default browser after startup",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> OperatorConfig:
    parser = build_parser()
    values = parser.parse_args(argv)
    allowed = {"all", "HRI", "Memory", "Planner"}
    invalid = [value for value in values.think if value not in allowed]
    if invalid:
        parser.error(
            "--think contains unsupported values: " + ", ".join(invalid)
        )
    if "all" in values.think and len(values.think) > 1:
        parser.error("--think all cannot be combined with agent names")
    return OperatorConfig(
        camera_base_url=values.camera_base_url,
        memory_store_path=values.memory_store_path,
        port=values.port,
        think=tuple(values.think),
        open_browser=values.open_browser,
    )


def build_runtime_factory(config: OperatorConfig) -> Callable[[], tuple[Any, Any]]:
    """Return a lazy vLLM runtime factory matching the local service layout."""

    def factory() -> tuple[Any, Any]:
        from prefmem.cli import parse_args as parse_agent_args
        from prefmem.runtime import build_runtime

        agent_argv = [
            "--model-config",
            "vllm",
            "--model-provider",
            "vllm",
            "--camera-base-url",
            config.camera_base_url,
            "--preference-store",
            str(config.memory_store_path),
            "--think",
            *config.think,
        ]
        return build_runtime(parse_agent_args(agent_argv))

    return factory


def _digest(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    rendered = " ".join(str(value).split())
    return rendered or fallback


def _timestamp(value: Any) -> str:
    raw = str(value or "")
    if "T" in raw:
        clock = raw.split("T", 1)[1].split("+", 1)[0].rstrip("Z")
        return clock[:8]
    return raw


def _list_text(values: Any, *keys: str) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
            continue
        if isinstance(value, Mapping):
            candidate = next(
                (value.get(key) for key in keys if value.get(key)),
                None,
            )
            if candidate is not None:
                result.append(_text(candidate))
    return result


def _safe_json(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
        value = decoded
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


@contextmanager
def _surface(classes: str = ""):
    with ui.card().classes(f"rp-surface {classes}".strip()) as card:
        yield card


class OperatorDashboard:
    """Per-browser projection of the process-wide :class:`AgentConsole`."""

    def __init__(self, console: AgentConsole, *, camera_base_url: str) -> None:
        self.console = console
        self.camera_base_url = _http_base_url(camera_base_url)
        self._digests: dict[str, str] = {}
        self._refreshing = False
        self._action_pending = False
        self._pending_label: str | None = None
        self._chat_follow = True
        self._live_turn_id: str | None = None
        self._live_revision = -1
        self._live_answer: Any | None = None
        self._live_reasoning: Any | None = None
        self._live_tools: Any | None = None
        self._live_trace: Any | None = None
        self._live_tools_digest: str | None = None

    def build(self) -> None:
        ui.colors(
            primary="#4f46e5",
            secondary="#0ea5e9",
            accent="#7c3aed",
            positive="#059669",
            negative="#dc2626",
            info="#0284c7",
            warning="#d97706",
        )
        ui.add_css(APP_CSS)

        with ui.header(elevated=False).classes("rp-header"):
            with ui.row().classes("rp-header-inner items-center no-wrap"):
                with ui.row().classes("items-center no-wrap gap-3 min-w-0"):
                    with ui.row().classes(
                        "rp-brand-mark items-center justify-center"
                    ):
                        ui.icon("smart_toy", size="23px")
                    with ui.column().classes("rp-brand-copy gap-0 min-w-0"):
                        ui.label("RoboPref").classes("rp-brand-title")
                        ui.label("Agent workspace").classes("rp-brand-subtitle")
                ui.space()
                self.health_header = ui.row().classes(
                    "rp-health-row items-center no-wrap"
                )
                self.state_header = ui.row().classes("items-center no-wrap")
                self.stop_button = ui.button(
                    "Stop",
                    icon="emergency",
                    on_click=self.stop_dialog_open,
                ).props("unelevated").classes("rp-header-stop")

        self.busy_bar = ui.linear_progress(
            value=0,
            show_value=False,
            color="primary",
        ).props("indeterminate").classes("rp-busy-bar")

        with ui.element("main").classes("rp-workspace"):
            with ui.column().classes("rp-left-pane"):
                self._build_camera()
                self._build_monitor()
                self._build_agent_grid()
            self._build_chat()

        self._build_dialogs()
        self._render(self.console.cached_snapshot(), force=True)
        ui.timer(0.25, self.refresh, immediate=False)

    def _build_camera(self) -> None:
        with _surface("rp-camera-card"):
            with ui.row().classes("rp-section-header items-center no-wrap"):
                with ui.row().classes("rp-section-icon camera items-center justify-center"):
                    ui.icon("videocam", size="20px")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label("Live workspace").classes("rp-eyebrow")
                    ui.label(
                        urlsplit(self.camera_base_url).netloc
                    ).classes("rp-section-title")
                ui.space()
                snapshot_link = ui.link(
                    "Snapshot",
                    target=f"{self.camera_base_url}/snapshot.jpg",
                    new_tab=True,
                ).classes("rp-quiet-link")
                snapshot_link.props["rel"] = "noopener noreferrer"
                snapshot_link.props["aria-label"] = (
                    "Open the current camera frame in a new tab"
                )
            with ui.element("div").classes("rp-camera-stage"):
                camera_feed = ui.element("img").classes("rp-camera-feed")
                camera_feed.props["src"] = (
                    f"{self.camera_base_url}/stream.mjpg"
                )
                camera_feed.props["alt"] = "Live workspace camera"
                self.camera_live_badge = ui.row().classes(
                    "rp-camera-badge unknown items-center no-wrap"
                )
                with self.camera_live_badge:
                    ui.element("span").classes("rp-status-dot")
                    self.camera_live_label = ui.label("Checking camera")
            with ui.row().classes("rp-camera-footer items-center no-wrap"):
                ui.icon("visibility", size="17px")
                self.camera_caption = ui.label("Waiting for runtime state…")
                ui.space()
                ui.label("MJPEG").classes("rp-mono")

    def _build_monitor(self) -> None:
        with _surface("rp-monitor-card"):
            with ui.row().classes("rp-section-header items-center no-wrap"):
                with ui.row().classes(
                    "rp-section-icon monitor items-center justify-center"
                ):
                    ui.icon("monitor_heart", size="20px")
                with ui.column().classes("gap-0"):
                    ui.label("Primary observer").classes("rp-eyebrow")
                    ui.label("Monitor agent").classes("rp-section-title")
                ui.space()
                self.monitor_status = ui.row().classes(
                    "rp-agent-status idle items-center no-wrap"
                )
                with self.monitor_status:
                    ui.element("span").classes("rp-status-dot")
                    self.monitor_status_label = ui.label("Standby")
            self.monitor_panel = ui.column().classes(
                "rp-monitor-body w-full"
            )

    def _build_agent_grid(self) -> None:
        with ui.column().classes("rp-agent-section w-full gap-2"):
            with ui.row().classes("items-center w-full"):
                ui.label("Other agents").classes("rp-group-title")
                ui.space()
                ui.label("Live working status").classes("rp-muted")
            self.agent_grid = ui.element("div").classes("rp-agent-grid")

    def _build_chat(self) -> None:
        with _surface("rp-chat-panel"):
            with ui.row().classes("rp-chat-header items-center no-wrap"):
                with ui.row().classes(
                    "rp-chat-avatar items-center justify-center"
                ):
                    ui.icon("forum", size="21px")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label("Conversation").classes("rp-chat-title")
                    self.chat_status = ui.label("Ready").classes(
                        "rp-chat-subtitle"
                    )
                ui.space()
                self.reset_button = ui.button(
                    "New chat",
                    icon="restart_alt",
                    on_click=self.reset_dialog_open,
                ).props("flat no-caps").classes("rp-new-chat")

            self.context_panel = ui.column().classes(
                "rp-context-slot w-full"
            )

            def on_scroll(event: Any) -> None:
                percentage = float(
                    getattr(event, "vertical_percentage", 1.0) or 0.0
                )
                self._chat_follow = percentage >= 0.97
                self.jump_button.set_visibility(not self._chat_follow)

            with ui.scroll_area(on_scroll=on_scroll).classes(
                "rp-chat-scroll"
            ) as scroll:
                self.chat_scroll = scroll
                self.chat_history = ui.column().classes(
                    "rp-chat-history w-full"
                )
                self.live_slot = ui.column().classes("rp-live-slot w-full")

            self.jump_button = ui.button(
                "Jump to latest",
                icon="arrow_downward",
                on_click=self.jump_to_latest,
            ).props("dense no-caps").classes("rp-jump-button")
            self.jump_button.set_visibility(False)

            with ui.row().classes("rp-composer items-end no-wrap"):
                self.chat_input = ui.input(
                    placeholder="Describe a goal, preference, or correction…"
                ).props("borderless dense clearable").classes(
                    "rp-chat-input flex-grow"
                )
                self.chat_input.on("keydown.enter", self.send_message)
                self.send_button = ui.button(
                    icon="arrow_upward",
                    on_click=self.send_message,
                ).props("round unelevated").classes("rp-send-button")
                self.send_button.props["aria-label"] = "Send message"
                self.send_button.tooltip("Send message")

    def _build_dialogs(self) -> None:
        with ui.dialog().classes("rp-dialog") as replan_dialog:
            self.replan_dialog = replan_dialog
            with ui.card().classes("rp-dialog-card"):
                ui.label("Request a fresh plan").classes("rp-dialog-title")
                ui.label(
                    "The confirmed high-level goal stays frozen. Add any new "
                    "observation or route guidance for the next cycle."
                ).classes("rp-dialog-copy")
                self.replan_guidance = ui.textarea(
                    label="Optional guidance",
                    placeholder="For example: approach from the left side…",
                ).props("outlined autogrow").classes("w-full")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button(
                        "Cancel", on_click=replan_dialog.close
                    ).props("flat no-caps")
                    ui.button(
                        "Replan",
                        icon="route",
                        on_click=self.request_replan,
                    ).props("unelevated no-caps").classes("rp-primary-action")

        with ui.dialog().classes("rp-dialog") as stop_dialog:
            self.stop_dialog = stop_dialog
            with ui.card().classes("rp-dialog-card"):
                with ui.row().classes("items-center no-wrap gap-3"):
                    with ui.row().classes(
                        "rp-danger-icon items-center justify-center"
                    ):
                        ui.icon("emergency", size="25px")
                    ui.label("Latch software stop?").classes("rp-dialog-title")
                ui.label(
                    "PrefMem will stop publishing work and this runtime cannot "
                    "be resumed. The current robot integration only logs the "
                    "stop request."
                ).classes("rp-dialog-copy")
                self.stop_reason = ui.input(
                    label="Reason",
                    value="Operator observed a potential hazard.",
                ).props("outlined").classes("w-full")
                ui.label(
                    "Not safety-rated: use the independent hardware emergency "
                    "stop whenever physical safety is at risk."
                ).classes("rp-safety-copy")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button(
                        "Cancel", on_click=stop_dialog.close
                    ).props("flat no-caps")
                    ui.button(
                        "Latch software stop",
                        icon="emergency",
                        on_click=self.latch_stop,
                    ).props("unelevated no-caps").classes("rp-danger-action")

        with ui.dialog().classes("rp-dialog") as reset_dialog:
            self.reset_dialog = reset_dialog
            with ui.card().classes("rp-dialog-card"):
                with ui.row().classes("items-center no-wrap gap-3"):
                    with ui.row().classes(
                        "rp-reset-icon items-center justify-center"
                    ):
                        ui.icon("restart_alt", size="24px")
                    ui.label("Start a new chat?").classes("rp-dialog-title")
                ui.label(
                    "This clears the shared conversation and starts a fresh HRI "
                    "checkpoint for every open console. The physical goal, "
                    "current task, Monitor, and validation state are preserved."
                ).classes("rp-dialog-copy")
                ui.label(
                    "A chat reset is not a safety stop. Wait for the current "
                    "agent request to finish before starting a new chat."
                ).classes("rp-dialog-note")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button(
                        "Keep chat", on_click=reset_dialog.close
                    ).props("flat no-caps")
                    ui.button(
                        "Start new chat",
                        icon="restart_alt",
                        on_click=self.reset_chat,
                    ).props("unelevated no-caps").classes("rp-primary-action")

    def stop_dialog_open(self) -> None:
        self.stop_dialog.open()

    def reset_dialog_open(self) -> None:
        self.reset_dialog.open()

    def jump_to_latest(self) -> None:
        self._chat_follow = True
        self.jump_button.set_visibility(False)
        self.chat_scroll.scroll_to(percent=1, duration=0.18)

    async def refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._render(self.console.cached_snapshot())
        finally:
            self._refreshing = False

    def _begin_local_action(self, label: str) -> bool:
        if self._action_pending:
            ui.notify("An operator request is already in progress.", type="info")
            return False
        self._action_pending = True
        self._pending_label = label
        self._render(self.console.cached_snapshot())
        return True

    def _finish_local_action(self) -> None:
        self._action_pending = False
        self._pending_label = None
        self._render(self.console.cached_snapshot())

    async def _invoke_action(
        self,
        label: str,
        callback: Callable[..., dict[str, Any]],
        *args: Any,
    ) -> dict[str, Any] | None:
        if not self._begin_local_action(label):
            return None
        try:
            return await run.io_bound(callback, *args)
        except Exception as error:
            ui.notify(
                f"The operator request failed: {_text(error)}",
                type="negative",
                multi_line=True,
                close_button=True,
            )
            return None
        finally:
            self._finish_local_action()

    async def send_message(self) -> None:
        value = self.chat_input.value
        if not isinstance(value, str) or not value.strip():
            ui.notify("Enter a message before sending.", type="warning")
            return
        self._chat_follow = True
        self.jump_button.set_visibility(False)
        self.chat_scroll.scroll_to(percent=1, duration=0.08)
        result = await self._invoke_action(
            "HRI is responding",
            self.console.run_turn,
            value,
        )
        if result is None:
            return
        if result["ok"]:
            self.chat_input.value = ""
            self.chat_input.update()
            self.jump_to_latest()
        else:
            ui.notify(
                result["message"],
                type="negative",
                multi_line=True,
                close_button=True,
            )

    async def start_runtime(self) -> None:
        result = await self._invoke_action(
            "Starting PrefMem",
            self.console.start,
        )
        if result is not None:
            self._notify_result(result)

    async def confirm_goal(self, goal_id: str, revision: int) -> None:
        result = await self._invoke_action(
            "Confirming goal",
            self.console.confirm_goal,
            goal_id,
            revision,
        )
        if result is not None:
            self._notify_result(result)

    async def decline_goal(self, goal_id: str, revision: int) -> None:
        result = await self._invoke_action(
            "Declining goal",
            self.console.decline_goal,
            goal_id,
            revision,
        )
        if result is not None:
            self._notify_result(result, success_type="info")

    async def resume_task(self) -> None:
        result = await self._invoke_action(
            "Resuming task",
            self.console.resume,
        )
        if result is not None:
            self._notify_result(result)

    async def request_replan(self) -> None:
        guidance = self.replan_guidance.value
        self.replan_dialog.close()
        result = await self._invoke_action(
            "Requesting a new plan",
            self.console.replan,
            guidance,
        )
        if result is None:
            return
        if result["ok"]:
            self.replan_guidance.value = ""
            self.replan_guidance.update()
        self._notify_result(result)

    async def latch_stop(self) -> None:
        reason = self.stop_reason.value
        self.stop_dialog.close()
        result = await run.io_bound(self.console.emergency_stop, reason)
        if not isinstance(result, Mapping):
            return
        self._render(self.console.cached_snapshot())
        ui.notify(
            result["message"],
            type="negative" if result["ok"] else "warning",
            multi_line=True,
            close_button=True,
            timeout=0,
        )

    async def reset_chat(self) -> None:
        self.reset_dialog.close()
        result = await self._invoke_action(
            "Resetting chat",
            self.console.reset_chat_session,
        )
        if not isinstance(result, Mapping):
            return
        if result.get("ok"):
            self._live_turn_id = None
            self._live_revision = -1
            self._live_tools_digest = None
            self._render(self.console.cached_snapshot())
            self.jump_to_latest()
            payload = result.get("payload")
            cleanup_failed = (
                isinstance(payload, Mapping)
                and bool(payload.get("checkpoint_cleanup_error"))
            )
            self._notify_result(
                result,
                success_type="warning" if cleanup_failed else "positive",
            )
            return
        self._render(self.console.cached_snapshot())
        self._notify_result(result)

    @staticmethod
    def _notify_result(
        result: Mapping[str, Any],
        *,
        success_type: str = "positive",
    ) -> None:
        ui.notify(
            str(result.get("message") or "Operator request completed."),
            type=success_type if result.get("ok") else "negative",
            multi_line=not bool(result.get("ok")),
            close_button=not bool(result.get("ok")),
        )

    def _render(self, snapshot: dict[str, Any], *, force: bool = False) -> None:
        context = snapshot.get("context") or {}
        state = str(context.get("state") or "OFFLINE").upper()
        busy = bool(snapshot.get("busy"))
        interaction_busy = busy or self._action_pending
        emergency = bool(context.get("emergency_latched"))

        self.busy_bar.set_visibility(interaction_busy)
        busy_label = (
            snapshot.get("busy_action")
            or self._pending_label
            or "Agent request"
        )
        self.chat_status.set_text(
            str(busy_label) if interaction_busy else "Ready for a message"
        )
        if interaction_busy or snapshot.get("closed") or emergency:
            self.chat_input.disable()
            self.send_button.disable()
        else:
            self.chat_input.enable()
            self.send_button.enable()

        if interaction_busy or snapshot.get("closed"):
            self.reset_button.disable()
        else:
            self.reset_button.enable()

        if not snapshot.get("started") or emergency or snapshot.get("closed"):
            self.stop_button.disable()
        else:
            self.stop_button.enable()

        self._render_camera(snapshot, state)
        self._render_if_changed(
            "header",
            {
                "state": state,
                "health": snapshot.get("health"),
                "busy": interaction_busy,
            },
            lambda: self._render_header(snapshot, state, interaction_busy),
            force,
        )
        self._render_if_changed(
            "context",
            {
                "started": snapshot.get("started"),
                "closed": snapshot.get("closed"),
                "state": state,
                "busy": interaction_busy,
                "goal": context.get("goal"),
                "attention_reason": context.get("attention_reason"),
            },
            lambda: self._render_context(
                snapshot,
                state,
                interaction_busy,
            ),
            force,
        )
        self._render_if_changed(
            "monitor",
            {
                "state": state,
                "context": context,
                "monitor": (snapshot.get("services") or {}).get("monitor"),
                "agent": (snapshot.get("agents") or {}).get("monitor"),
            },
            lambda: self._render_monitor(snapshot, state),
            force,
        )
        self._render_if_changed(
            "agents",
            {
                "started": snapshot.get("started"),
                "state": state,
                "busy_action": snapshot.get("busy_action"),
                "agents": snapshot.get("agents"),
                "services": snapshot.get("services"),
            },
            lambda: self._render_agents(snapshot, state),
            force,
        )
        self._render_if_changed(
            "history",
            snapshot.get("transcript"),
            lambda: self._render_chat_history(snapshot),
            force,
        )
        self._paint_live_turn(snapshot, force=force)

    def _render_if_changed(
        self,
        name: str,
        value: Any,
        renderer: Callable[[], None],
        force: bool,
    ) -> None:
        current = _digest(value)
        if not force and self._digests.get(name) == current:
            return
        renderer()
        self._digests[name] = current

    def _render_camera(self, snapshot: dict[str, Any], state: str) -> None:
        camera = (snapshot.get("health") or {}).get("camera") or {}
        status = str(camera.get("status") or "unknown")
        if status not in {"online", "degraded"}:
            status = "unknown"
        self.camera_live_badge.classes(
            add=status,
            remove="online degraded unknown",
        )
        self.camera_live_label.set_text(
            "Live"
            if status == "online"
            else "Unavailable"
            if status == "degraded"
            else "Checking"
        )
        state_label = STATE_META.get(state, (state, "circle", "offline"))[0]
        cycle = (snapshot.get("context") or {}).get("cycle_id") or 0
        self.camera_caption.set_text(f"{state_label} · cycle {cycle}")

    def _render_header(
        self,
        snapshot: dict[str, Any],
        state: str,
        interaction_busy: bool,
    ) -> None:
        health = snapshot.get("health") or {}
        labels = {"camera": "Camera", "gemma": "Gemma", "embedding": "Embed"}
        self.health_header.clear()
        with self.health_header:
            for name in ("camera", "gemma", "embedding"):
                record = health.get(name) or {}
                status = str(record.get("status") or "unknown")
                if status not in {"online", "degraded"}:
                    status = "unknown"
                detail = record.get("error") or (
                    "Online" if record.get("ok") else "Not checked"
                )
                pill = ui.row().classes(
                    f"rp-health-pill {status} items-center no-wrap"
                )
                pill.props["aria-label"] = f"{labels[name]}: {_text(detail)}"
                with pill:
                    ui.element("span").classes("rp-status-dot")
                    ui.label(labels[name]).classes("rp-health-label")
                    ui.tooltip(_text(detail))

        label, icon, status_class = STATE_META.get(
            state,
            (state, "circle", "offline"),
        )
        self.state_header.clear()
        with self.state_header:
            with ui.row().classes(
                f"rp-runtime-chip {status_class} items-center no-wrap"
            ):
                if interaction_busy:
                    ui.spinner("dots", size="15px", color="primary")
                else:
                    ui.icon(icon, size="16px")
                ui.label(label)

    def _render_context(
        self,
        snapshot: dict[str, Any],
        state: str,
        interaction_busy: bool,
    ) -> None:
        context = snapshot.get("context") or {}
        self.context_panel.clear()
        with self.context_panel:
            if snapshot.get("closed"):
                self._context_callout(
                    "Console closed",
                    "This process no longer accepts operator requests.",
                    "lock",
                    "danger",
                )
                return
            if not snapshot.get("started"):
                with ui.row().classes(
                    "rp-context-banner neutral items-center no-wrap"
                ):
                    ui.icon("power_settings_new", size="19px")
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label("Runtime is offline").classes(
                            "rp-context-title"
                        )
                        ui.label(
                            "Sending a message also starts PrefMem automatically."
                        ).classes("rp-context-copy")
                    ui.space()
                    start = ui.button(
                        "Start",
                        icon="play_arrow",
                        on_click=self.start_runtime,
                    ).props("unelevated no-caps").classes("rp-primary-action")
                    if interaction_busy:
                        start.disable()
                return
            if state == "AWAITING_CONFIRMATION":
                goal = context.get("goal")
                if not isinstance(goal, Mapping):
                    return
                with ui.column().classes("rp-context-banner attention w-full"):
                    with ui.row().classes("items-start no-wrap w-full"):
                        ui.icon("fact_check", size="20px")
                        with ui.column().classes("gap-1 min-w-0 flex-grow"):
                            ui.label("Confirm the exact goal").classes(
                                "rp-context-title"
                            )
                            ui.label(goal.get("goal") or "Unnamed goal").classes(
                                "rp-context-goal"
                            )
                            constraints = _list_text(goal.get("constraints"))
                            if constraints:
                                ui.label(
                                    "Constraints · " + " · ".join(constraints)
                                ).classes("rp-context-copy")
                    goal_id = goal.get("goal_id")
                    revision = goal.get("revision")
                    valid = isinstance(goal_id, str) and type(revision) is int

                    async def confirm_exact() -> None:
                        if valid:
                            await self.confirm_goal(goal_id, revision)

                    async def decline_exact() -> None:
                        if valid:
                            await self.decline_goal(goal_id, revision)

                    with ui.row().classes("w-full justify-end gap-2"):
                        decline = ui.button(
                            "Decline",
                            icon="close",
                            on_click=decline_exact,
                        ).props("flat no-caps")
                        confirm = ui.button(
                            "Confirm & execute",
                            icon="play_arrow",
                            on_click=confirm_exact,
                        ).props("unelevated no-caps").classes(
                            "rp-primary-action"
                        )
                        if interaction_busy or not valid:
                            decline.disable()
                            confirm.disable()
                return
            if state == "NEEDS_ATTENTION":
                with ui.row().classes(
                    "rp-context-banner attention items-center no-wrap"
                ):
                    ui.icon("report_problem", size="20px")
                    with ui.column().classes("gap-0 min-w-0 flex-grow"):
                        ui.label("Execution needs attention").classes(
                            "rp-context-title"
                        )
                        ui.label(
                            context.get("attention_reason")
                            or "Choose how PrefMem should recover."
                        ).classes("rp-context-copy")
                    resume = ui.button(
                        "Resume",
                        icon="replay",
                        on_click=self.resume_task,
                    ).props("flat no-caps")
                    replan = ui.button(
                        "Replan",
                        icon="route",
                        on_click=self.replan_dialog.open,
                    ).props("unelevated no-caps").classes("rp-primary-action")
                    if interaction_busy:
                        resume.disable()
                        replan.disable()
                return
            if state == "EMERGENCY_STOPPED":
                self._context_callout(
                    "Software stop is latched",
                    "Restart the process only after the scene is safe.",
                    "emergency",
                    "danger",
                )
                return
            if state == "COMPLETE":
                self._context_callout(
                    "Goal complete",
                    "The frozen validation checklist has been satisfied.",
                    "task_alt",
                    "complete",
                )

    @staticmethod
    def _context_callout(
        title: str,
        copy: str,
        icon: str,
        style: str,
    ) -> None:
        with ui.row().classes(
            f"rp-context-banner {style} items-center no-wrap"
        ):
            ui.icon(icon, size="20px")
            with ui.column().classes("gap-0 min-w-0"):
                ui.label(title).classes("rp-context-title")
                ui.label(copy).classes("rp-context-copy")

    def _render_monitor(self, snapshot: dict[str, Any], state: str) -> None:
        context = snapshot.get("context") or {}
        task = context.get("current_task")
        services = snapshot.get("services") or {}
        monitor_service = services.get("monitor") or {}
        agent_record = (snapshot.get("agents") or {}).get("monitor") or {}

        if state == "EMERGENCY_STOPPED":
            monitor_label, monitor_class = "Stopped", "danger"
        elif not snapshot.get("started"):
            monitor_label, monitor_class = "Offline", "offline"
        elif str(agent_record.get("status") or "").lower() in {
            "working",
            "running",
            "watching",
        }:
            monitor_label, monitor_class = "Working", "working"
        elif monitor_service.get("running"):
            monitor_label, monitor_class = "Standby", "ready"
        else:
            monitor_label, monitor_class = "Idle", "idle"

        self.monitor_status.classes(
            add=monitor_class,
            remove="working ready idle offline danger",
        )
        self.monitor_status_label.set_text(monitor_label)
        self.monitor_panel.clear()
        with self.monitor_panel:
            if not isinstance(task, Mapping) or str(task.get("phase")) != "STEP":
                with ui.row().classes("rp-monitor-empty items-center no-wrap"):
                    ui.icon("radar", size="28px")
                    with ui.column().classes("gap-0"):
                        ui.label("No step under observation").classes(
                            "rp-monitor-empty-title"
                        )
                        ui.label(
                            "Monitor will attach when the controller publishes a task."
                        ).classes("rp-muted")
                if context.get("latest_observation"):
                    self._monitor_observation(context["latest_observation"])
                return

            with ui.row().classes("rp-monitor-summary items-start no-wrap w-full"):
                with ui.column().classes("gap-1 min-w-0 flex-grow"):
                    ui.label("Current instruction").classes("rp-field-label")
                    ui.label(task.get("instruction") or "Unnamed task").classes(
                        "rp-monitor-instruction"
                    )
                with ui.element("div").classes("rp-monitor-cycle"):
                    ui.label("Cycle").classes("rp-field-label")
                    ui.label(str(task.get("cycle_id") or "—")).classes(
                        "rp-monitor-cycle-value"
                    )

            observation = (
                context.get("latest_observation")
                or task.get("monitor_observation")
                or "Waiting for a post-publication frame."
            )
            self._monitor_observation(observation)

            criteria = _list_text(
                task.get("expected_observation"),
                "description",
                "label",
            )
            if criteria:
                with ui.column().classes("rp-monitor-criteria w-full"):
                    ui.label("Visual success criteria").classes("rp-field-label")
                    for criterion in criteria[:3]:
                        with ui.row().classes("items-start no-wrap gap-2"):
                            ui.icon("check_circle_outline", size="16px")
                            ui.label(criterion).classes("rp-monitor-criterion")

            if context.get("attention_reason"):
                with ui.row().classes(
                    "rp-monitor-warning items-start no-wrap"
                ):
                    ui.icon("warning_amber", size="18px")
                    ui.label(context["attention_reason"])

    @staticmethod
    def _monitor_observation(observation: Any) -> None:
        with ui.row().classes("rp-observation items-start no-wrap w-full"):
            ui.icon("visibility", size="18px")
            with ui.column().classes("gap-0 min-w-0"):
                ui.label("Latest observation").classes("rp-field-label")
                ui.label(_text(observation)).classes("rp-observation-copy")

    def _render_agents(self, snapshot: dict[str, Any], state: str) -> None:
        records = snapshot.get("agents") or {}
        fallback = self._fallback_agent_statuses(snapshot, state)
        self.agent_grid.clear()
        with self.agent_grid:
            for key, (label, icon) in AGENT_META.items():
                record = records.get(key) or {}
                raw_status = str(
                    record.get("status") or fallback[key]
                ).lower()
                status_class, status_label = self._agent_status_meta(raw_status)
                with ui.card().classes("rp-mini-agent"):
                    with ui.row().classes(
                        "rp-mini-agent-row items-center no-wrap w-full"
                    ):
                        with ui.row().classes(
                            "rp-mini-agent-icon items-center justify-center"
                        ):
                            ui.icon(icon, size="18px")
                        ui.label(label).classes("rp-mini-agent-name")
                        ui.space()
                        with ui.row().classes(
                            f"rp-agent-status {status_class} items-center no-wrap"
                        ):
                            ui.element("span").classes("rp-status-dot")
                            ui.label(status_label)

    @staticmethod
    def _fallback_agent_statuses(
        snapshot: Mapping[str, Any],
        state: str,
    ) -> dict[str, str]:
        if not snapshot.get("started"):
            return {name: "offline" for name in AGENT_META}
        if state == "EMERGENCY_STOPPED":
            return {name: "stopped" for name in AGENT_META}
        busy_action = str(snapshot.get("busy_action") or "").lower()
        services = snapshot.get("services") or {}
        validator = services.get("validator") or {}
        return {
            "hri": "working" if "hri" in busy_action else "ready",
            "planner": "working" if state == "PLANNING" else "standby",
            "memory": "standby",
            "validator": (
                "watching"
                if validator.get("active_publication_id")
                else "standby"
            ),
        }

    @staticmethod
    def _agent_status_meta(status: str) -> tuple[str, str]:
        if status in {
            "working",
            "running",
            "starting",
            "active",
            "monitoring",
            "watching",
        }:
            if status == "starting":
                return "working", "Starting"
            return "working", "Working" if status != "watching" else "Watching"
        if status in {"ready", "online"}:
            return "ready", "Ready"
        if status in {"stopped", "error", "degraded", "danger"}:
            return "danger", "Stopped" if status == "stopped" else "Issue"
        if status == "offline":
            return "offline", "Offline"
        return "idle", "Standby"

    def _render_chat_history(self, snapshot: dict[str, Any]) -> None:
        transcript = snapshot.get("transcript") or []
        trace = snapshot.get("trace") or []
        self.chat_history.clear()
        with self.chat_history:
            if not transcript:
                with ui.column().classes("rp-chat-empty items-center"):
                    with ui.row().classes(
                        "rp-empty-orb items-center justify-center"
                    ):
                        ui.icon("waving_hand", size="28px")
                    ui.label("What should the robot accomplish?").classes(
                        "rp-empty-title"
                    )
                    ui.label(
                        "Describe the physical outcome. RoboPref will clarify "
                        "preferences and show the exact goal before execution."
                    ).classes("rp-empty-copy")
            for message in transcript[-100:]:
                if not isinstance(message, Mapping):
                    continue
                role = str(message.get("role") or "assistant")
                if role == "user":
                    self._render_user_message(message)
                else:
                    turn_id = message.get("turn_id")
                    turn_trace = [
                        event
                        for event in trace
                        if isinstance(event, Mapping)
                        and turn_id
                        and event.get("turn_id") == turn_id
                    ]
                    self._render_assistant_message(message, turn_trace)
        if transcript:
            if self._chat_follow:
                self.chat_scroll.scroll_to(percent=1, duration=0.12)
            else:
                self.jump_button.set_visibility(True)

    @staticmethod
    def _render_user_message(message: Mapping[str, Any]) -> None:
        with ui.row().classes("rp-message-row user w-full justify-end"):
            with ui.column().classes("rp-message-stack user items-end"):
                ui.label(str(message.get("content") or "")).classes(
                    "rp-message-bubble user whitespace-pre-wrap"
                )
                ui.label(_timestamp(message.get("timestamp"))).classes(
                    "rp-message-time"
                )

    def _render_assistant_message(
        self,
        message: Mapping[str, Any],
        trace: list[Mapping[str, Any]],
    ) -> None:
        with ui.row().classes("rp-message-row assistant w-full items-start"):
            with ui.row().classes(
                "rp-message-avatar items-center justify-center"
            ):
                ui.icon("smart_toy", size="17px")
            with ui.column().classes("rp-message-stack assistant"):
                self._render_trace(trace)
                content = str(message.get("content") or "")
                bubble_class = " error" if message.get("error") else ""
                with ui.element("div").classes(
                    f"rp-message-bubble assistant{bubble_class}"
                ):
                    ui.markdown(content, sanitize=True)
                ui.label(_timestamp(message.get("timestamp"))).classes(
                    "rp-message-time"
                )

    def _render_trace(self, trace: list[Mapping[str, Any]]) -> None:
        sections = self._trace_sections(trace)
        if not sections:
            return
        tool_count = sum(
            1 for title, _node, _content in sections if title.startswith("Tool")
        )
        title = "Internal activity"
        if tool_count:
            title += f" · {tool_count} tool event{'s' if tool_count != 1 else ''}"
        with ui.expansion(title, icon="terminal", value=False).classes(
            "rp-trace-expansion"
        ):
            with ui.column().classes("rp-trace-list w-full"):
                for section_title, node, content in sections:
                    with ui.column().classes("rp-trace-item w-full"):
                        with ui.row().classes("items-center no-wrap w-full"):
                            ui.label(section_title).classes("rp-trace-kind")
                            ui.space()
                            ui.label(node or "HRI Agent").classes(
                                "rp-trace-node"
                            )
                        ui.label(content).classes(
                            "rp-trace-content whitespace-pre-wrap"
                        )

    @staticmethod
    def _trace_sections(
        trace: Iterable[Mapping[str, Any]],
    ) -> list[tuple[str, str, str]]:
        reasoning: list[str] = []
        tool_deltas: list[str] = []
        completed_calls: list[tuple[str, str, str]] = []
        results: list[tuple[str, str, str]] = []
        outputs: list[tuple[str, str, str]] = []
        for event in trace:
            kind = str(event.get("kind") or "")
            node = str(event.get("node") or "")
            content = str(event.get("content") or "")
            if kind == "thinking_delta":
                reasoning.append(content)
            elif kind == "tool_call_delta":
                tool_deltas.append(content)
            elif kind == "tool_call":
                completed_calls.append(
                    ("Tool call", node, content or _safe_json(event.get("payload")))
                )
            elif kind == "tool_result":
                results.append(
                    ("Tool result", node, content or _safe_json(event.get("payload")))
                )
            elif kind not in {
                "assistant_delta",
                "turn_start",
                "turn_complete",
            } and content:
                outputs.append(("Agent output", node, content))
        sections: list[tuple[str, str, str]] = []
        if reasoning:
            sections.append(("Thinking", "HRI Agent", "".join(reasoning)))
        if completed_calls:
            sections.extend(completed_calls)
        elif tool_deltas:
            sections.append(("Tool call", "HRI Agent", "".join(tool_deltas)))
        sections.extend(results)
        sections.extend(outputs)
        return [section for section in sections if section[2]]

    def _paint_live_turn(
        self,
        snapshot: Mapping[str, Any],
        *,
        force: bool,
    ) -> None:
        active = snapshot.get("active_turn")
        active_status = (
            str(active.get("status") or "").lower()
            if isinstance(active, Mapping)
            else ""
        )
        if not isinstance(active, Mapping) or active_status not in {
            "running",
            "streaming",
        }:
            if self._live_turn_id is not None or force:
                self.live_slot.clear()
                self._live_turn_id = None
                self._live_revision = -1
                self._live_tools_digest = None
            return
        turn_id = str(active.get("turn_id") or "active")
        revision = int(active.get("revision") or 0)
        if turn_id != self._live_turn_id:
            self._build_live_turn(turn_id)
        if not force and revision == self._live_revision:
            return
        self._live_revision = revision

        answer = str(active.get("assistant_text") or "")
        reasoning = str(active.get("reasoning_text") or "")
        tools = active.get("tools") or []
        if self._live_answer is not None:
            self._live_answer.set_text(answer or "HRI is working…")
        if self._live_reasoning is not None:
            self._live_reasoning.set_text(reasoning or "Waiting for model output…")
        if self._live_trace is not None:
            self._live_trace.set_text(
                "Internal activity"
                + (f" · {len(tools)} tools" if tools else "")
            )
        tools_digest = _digest(tools)
        if (
            self._live_tools is not None
            and tools_digest != self._live_tools_digest
        ):
            self._live_tools_digest = tools_digest
            self._live_tools.clear()
            with self._live_tools:
                for tool in tools:
                    if not isinstance(tool, Mapping):
                        continue
                    with ui.column().classes("rp-trace-item w-full"):
                        ui.label(
                            str(tool.get("name") or "Tool call")
                        ).classes("rp-trace-kind")
                        detail = (
                            tool.get("result")
                            or tool.get("arguments")
                            or tool.get("args")
                            or tool
                        )
                        ui.label(_safe_json(detail)).classes(
                            "rp-trace-content whitespace-pre-wrap"
                        )
        if self._chat_follow:
            self.chat_scroll.scroll_to(percent=1, duration=0.08)
        else:
            self.jump_button.set_visibility(True)

    def _build_live_turn(self, turn_id: str) -> None:
        self.live_slot.clear()
        self._live_turn_id = turn_id
        self._live_revision = -1
        self._live_tools_digest = None
        with self.live_slot:
            with ui.row().classes(
                "rp-message-row assistant live w-full items-start"
            ):
                with ui.row().classes(
                    "rp-message-avatar working items-center justify-center"
                ):
                    ui.spinner("dots", size="15px", color="primary")
                with ui.column().classes("rp-message-stack assistant"):
                    self._live_trace = ui.expansion(
                        "Internal activity",
                        icon="terminal",
                        value=False,
                    ).classes("rp-trace-expansion")
                    with self._live_trace:
                        with ui.column().classes("rp-trace-list w-full"):
                            with ui.column().classes("rp-trace-item w-full"):
                                ui.label("Thinking").classes("rp-trace-kind")
                                self._live_reasoning = ui.label(
                                    "Waiting for model output…"
                                ).classes(
                                    "rp-trace-content whitespace-pre-wrap"
                                )
                            self._live_tools = ui.column().classes("w-full gap-2")
                    with ui.element("div").classes(
                        "rp-message-bubble assistant streaming"
                    ):
                        self._live_answer = ui.label("HRI is working…").classes(
                            "whitespace-pre-wrap"
                        )


def register_operator_page(
    console: AgentConsole,
    *,
    camera_base_url: str = DEFAULT_CAMERA_BASE_URL,
) -> Callable[[], None]:
    """Register the singleton console projection and process maintenance."""

    camera_origin = _http_base_url(camera_base_url)

    async def maintain_runtime() -> None:
        await run.io_bound(console.tick)

    async def maintain_health() -> None:
        await run.io_bound(console.probe_health)

    app.timer(0.75, maintain_runtime, immediate=False)
    app.timer(15.0, maintain_health, immediate=True)

    @ui.page(
        "/",
        title="RoboPref Agent Workspace",
        dark=False,
        viewport="width=device-width, initial-scale=1",
        response_timeout=10.0,
    )
    def operator_page() -> None:
        OperatorDashboard(
            console,
            camera_base_url=camera_origin,
        ).build()

    return operator_page


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    health_endpoints = {
        "camera": f"{config.camera_base_url}/healthz",
        "gemma": "http://localhost:8000/v1/models",
        "embedding": "http://localhost:8080/v1/models",
    }
    console = AgentConsole(
        runtime_factory=build_runtime_factory(config),
        health_endpoints=health_endpoints,
    )
    register_operator_page(console, camera_base_url=config.camera_base_url)
    app.on_shutdown(console.close)
    try:
        ui.run(
            host=DEFAULT_GUI_HOST,
            port=config.port,
            title="RoboPref Agent Workspace",
            dark=False,
            show=config.open_browser,
            reload=False,
            uvicorn_logging_level="warning",
            endpoint_documentation="none",
            show_welcome_message=False,
        )
    except KeyboardInterrupt:
        pass


__all__ = [
    "DEFAULT_CAMERA_BASE_URL",
    "DEFAULT_GUI_HOST",
    "DEFAULT_GUI_PORT",
    "OperatorConfig",
    "OperatorDashboard",
    "build_parser",
    "build_runtime_factory",
    "main",
    "parse_args",
    "register_operator_page",
]
