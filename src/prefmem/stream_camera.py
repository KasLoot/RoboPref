"""Stream a local webcam to a web browser as an MJPEG feed."""

from __future__ import annotations

import argparse
import copy
import importlib
import ipaddress
import json
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import NoReturn
from urllib.parse import parse_qs, urlsplit

from prefmem.task_publisher import (
    ControllerDisplay,
    DisplayState,
    FRAME_SEQUENCE_HEADER,
    TASK_API_PATH,
)


PAGE = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Webcam stream</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body {
      display: grid;
      min-height: 100vh;
      margin: 0;
      place-items: center;
      background: #101418;
    }
    main { width: min(96vw, 1280px); }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 0.75rem 0;
    }
    h1 { margin: 0; font-size: 1.15rem; font-weight: 600; }
    #status { color: #f5c451; font-size: 0.9rem; }
    #status.online { color: #65d897; }
    img {
      display: block;
      width: 100%;
      min-height: 12rem;
      border-radius: 0.75rem;
      background: #050708;
      box-shadow: 0 1rem 3rem #0008;
      object-fit: contain;
    }
    #task-card {
      margin-top: 0.9rem;
      padding: 1rem 1.1rem;
      border: 1px solid #2d3740;
      border-radius: 0.75rem;
      background: #171d22;
      box-shadow: 0 0.5rem 1.5rem #0004;
    }
    #task-card header { margin: 0; }
    #task-card h2 { margin: 0; font-size: 0.9rem; color: #9da7b0; }
    #task-cycle { color: #9da7b0; font-size: 0.85rem; }
    #task-status {
      display: inline-block;
      margin-top: 0.65rem;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: #33404a;
      color: #dce3e8;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    #task-status[data-status="ACTIVE"] { background: #164f78; color: #b9e4ff; }
    #task-status[data-status="PLANNING"] { background: #3f4f5a; color: #e3e9ed; }
    #task-status[data-status="FINAL_VALIDATION"] {
      background: #49366e;
      color: #dacaff;
    }
    #task-status[data-status="COMPLETE"] { background: #175c39; color: #9ef0bf; }
    #task-status[data-status="EMERGENCY_STOPPED"] {
      background: #7b1e28;
      color: #ffd1d4;
    }
    #task-status[data-status="NEEDS_ATTENTION"] {
      background: #684f16;
      color: #ffe09a;
    }
    #task-instruction { margin: 0.7rem 0 0; font-size: 1.15rem; line-height: 1.45; }
    #expected-container { margin-top: 0.85rem; }
    #expected-container h3 {
      margin: 0;
      color: #9da7b0;
      font-size: 0.8rem;
      font-weight: 600;
    }
    #expected-observation { margin: 0.4rem 0 0; padding-left: 1.25rem; }
    #expected-observation li + li { margin-top: 0.25rem; }
    .task-detail { margin: 0.7rem 0 0; color: #c8d0d6; font-size: 0.9rem; }
    .task-detail strong { color: #9da7b0; }
    footer { margin-top: 0.75rem; color: #9da7b0; font-size: 0.8rem; }
    a { color: #8fc7ff; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Live webcam</h1>
      <span id="status">Connecting...</span>
    </header>
    <img id="feed" src="/stream.mjpg" alt="Live webcam feed">
    <footer><a href="/snapshot.jpg">Open current frame</a></footer>
    <section id="task-card" aria-live="polite">
      <header>
        <h2>Current task</h2>
        <span id="task-cycle"></span>
      </header>
      <span id="task-status" data-status="WAITING">WAITING</span>
      <p id="task-instruction">Waiting for a task to be published.</p>
      <p id="task-goal" class="task-detail" hidden></p>
      <section id="expected-container" hidden>
        <h3>Expected observation</h3>
        <ul id="expected-observation"></ul>
      </section>
      <p id="task-message" class="task-detail" hidden></p>
      <p id="monitor-observation" class="task-detail" hidden></p>
    </section>
  </main>
  <script>
    const feed = document.querySelector("#feed");
    const status = document.querySelector("#status");
    feed.addEventListener("load", () => {
      status.textContent = "Live";
      status.className = "online";
    });
    feed.addEventListener("error", () => {
      status.textContent = "Stream unavailable";
      status.className = "";
    });

    const taskStatus = document.querySelector("#task-status");
    const taskCycle = document.querySelector("#task-cycle");
    const taskInstruction = document.querySelector("#task-instruction");
    const taskGoal = document.querySelector("#task-goal");
    const expectedContainer = document.querySelector("#expected-container");
    const expectedObservation = document.querySelector("#expected-observation");
    const taskMessage = document.querySelector("#task-message");
    const monitorObservation = document.querySelector("#monitor-observation");
    let displayRevision = -1;

    function showDetail(element, label, value) {
      element.replaceChildren();
      element.hidden = !value;
      if (!value) return;
      const heading = document.createElement("strong");
      heading.textContent = `${label}: `;
      element.append(heading, document.createTextNode(value));
    }

    function renderDisplay(envelope) {
      if (envelope.revision === displayRevision) return;
      displayRevision = envelope.revision;
      const display = envelope.display;
      if (!display) {
        taskStatus.textContent = "WAITING";
        taskStatus.dataset.status = "WAITING";
        taskCycle.textContent = "";
        taskInstruction.textContent = "Waiting for a task to be published.";
        showDetail(taskGoal, "Goal", null);
        expectedContainer.hidden = true;
        expectedObservation.replaceChildren();
        showDetail(taskMessage, "Message", null);
        showDetail(monitorObservation, "Monitor", null);
        return;
      }

      taskStatus.textContent = display.state;
      taskStatus.dataset.status = display.state;
      taskCycle.textContent = display.cycle ? `Cycle ${display.cycle}` : "";
      if (display.state === "ACTIVE") {
        taskInstruction.textContent = display.instruction;
      } else if (display.state === "PLANNING") {
        taskInstruction.textContent = "Planning the next task. Hold position.";
      } else if (display.state === "FINAL_VALIDATION") {
        taskInstruction.textContent =
          "Validating the final goal. Keep scene objects unchanged; follow the guidance below.";
      } else if (display.state === "COMPLETE") {
        taskInstruction.textContent = "The high-level goal is complete.";
      } else if (display.state === "EMERGENCY_STOPPED") {
        taskInstruction.textContent = "Emergency stop activated. Do not continue.";
      } else {
        taskInstruction.textContent = "Execution is paused.";
      }
      showDetail(taskGoal, "Goal", display.goal);

      const expectations = display.expected_observation || [];
      expectedObservation.replaceChildren();
      for (const criterion of expectations) {
        const item = document.createElement("li");
        item.textContent = criterion;
        expectedObservation.append(item);
      }
      expectedContainer.hidden = expectations.length === 0;
      showDetail(taskMessage, "Message", display.message);
      showDetail(monitorObservation, "Monitor", display.monitor_observation);
    }

    async function refreshDisplay() {
      try {
        const response = await fetch("/api/task", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderDisplay(await response.json());
      } catch (error) {
        displayRevision = -1;
        taskStatus.textContent = "UNAVAILABLE";
        taskStatus.dataset.status = "UNAVAILABLE";
        taskCycle.textContent = "";
        taskInstruction.textContent = "Task display unavailable.";
      } finally {
        window.setTimeout(refreshDisplay, 1000);
      }
    }

    refreshDisplay();
  </script>
</body>
</html>
"""

SYSTEM_CHOICES = ("windows", "linux")
MAX_TASK_BODY_BYTES = 64 * 1024


def validate_display_payload(value: object) -> dict[str, object]:
    """Validate a complete single-slot controller display payload."""

    return ControllerDisplay.from_payload(value).to_payload()


# Transitional import compatibility for callers of the prototype API.
validate_task_payload = validate_display_payload


class DisplayConflictError(ValueError):
    """A display mutation conflicts with newer or terminal state."""


class ControllerDisplayStore:
    """Keep one fenced controller display for concurrent HTTP clients.

    A session can be replaced only by a new session.  Replaced/reset session
    IDs remain retired, which prevents an in-flight update from an earlier run
    taking ownership of the page later.  ``COMPLETE`` and
    ``EMERGENCY_STOPPED`` are immutable within their session; only a different
    session or an explicit reset can replace them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision = 0
        self._display: dict[str, object] | None = None
        self._retired_session_ids: set[str] = set()

    def publish(self, value: object) -> dict[str, object]:
        display = validate_display_payload(value)
        session_id = display["session_id"]
        sequence = display["sequence"]
        assert isinstance(session_id, str)
        assert isinstance(sequence, int)
        with self._lock:
            if session_id in self._retired_session_ids:
                raise DisplayConflictError(
                    f"session {session_id!r} has been retired"
                )
            current = self._display
            if current is not None and current["session_id"] == session_id:
                current_sequence = current["sequence"]
                assert isinstance(current_sequence, int)
                if sequence < current_sequence:
                    raise DisplayConflictError(
                        "display sequence is older than the current sequence"
                    )
                if sequence == current_sequence:
                    if display == current:
                        return self._snapshot_locked()
                    raise DisplayConflictError(
                        "display sequence was reused with different content"
                    )
                if current["state"] in {
                    DisplayState.COMPLETE.value,
                    DisplayState.EMERGENCY_STOPPED.value,
                }:
                    raise DisplayConflictError(
                        f"{current['state']} is terminal for this session"
                    )
            elif current is not None:
                previous_session = current["session_id"]
                assert isinstance(previous_session, str)
                self._retired_session_ids.add(previous_session)

            self._revision += 1
            self._display = copy.deepcopy(display)
            return self._snapshot_locked()

    def reset(self, *, session_id: str | None = None) -> dict[str, object]:
        """Clear the slot and retire the prior session against late writes."""

        if session_id is not None:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("session_id must be a non-empty string")
            if len(session_id) > 128:
                raise ValueError("session_id must be at most 128 characters")
            session_id = session_id.strip()
        with self._lock:
            if self._display is not None:
                current_session = self._display["session_id"]
                assert isinstance(current_session, str)
                if session_id is not None and session_id != current_session:
                    raise DisplayConflictError(
                        "reset session_id does not own the current display"
                    )
                self._retired_session_ids.add(current_session)
                self._display = None
                self._revision += 1
            elif session_id is not None:
                self._retired_session_ids.add(session_id)
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        return {
            "revision": self._revision,
            "display": copy.deepcopy(self._display),
        }


# Kept as a source-compatible name while callers move to ControllerDisplayStore.
CurrentTaskStore = ControllerDisplayStore


def is_loopback_address(address: str) -> bool:
    """Return whether an HTTP peer address is IPv4/IPv6 loopback."""
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        return parsed.ipv4_mapped.is_loopback
    return parsed.is_loopback


def load_opencv() -> ModuleType:
    """Load OpenCV with an actionable error when the dependency is absent."""
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise
        raise SystemExit(
            "OpenCV is not installed. Run `uv sync`, then try again."
        ) from exc


def parse_camera(value: str) -> int:
    """Parse a non-negative OpenCV camera index."""
    try:
        camera = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera must be an integer index") from exc
    if camera < 0:
        raise argparse.ArgumentTypeError("camera must be zero or greater")
    return camera


def bounded_integer(minimum: int, maximum: int):
    """Build an argparse parser for a bounded integer."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}"
            )
        return number

    return parse


def default_system() -> str:
    """Return the camera system matching the current Python platform."""
    return "windows" if sys.platform == "win32" else "linux"


def backend_id(cv2: ModuleType, system: str, name: str) -> int:
    """Translate system and backend choices to an OpenCV backend ID."""
    if system == "linux":
        return cv2.CAP_V4L2

    backends = {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }
    return backends[name]


def configure_capture(
    cv2: ModuleType,
    capture,
    system: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Configure the camera, requesting compressed transport on Linux."""
    if system == "linux":
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def fourcc_name(value: float) -> str:
    """Decode an OpenCV FOURCC property into a printable name."""
    encoded = int(value)
    name = "".join(chr((encoded >> (8 * index)) & 0xFF) for index in range(4))
    return name.rstrip("\x00") or "unknown"


class CameraStream:
    """Capture and encode frames once for any number of HTTP clients."""

    def __init__(
        self,
        cv2: ModuleType,
        camera: int,
        system: str,
        backend: int,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
    ) -> None:
        self.cv2 = cv2
        self.camera = camera
        self.system = system
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._capture = None
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._sequence = 0
        self._error: str | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        capture = self.cv2.VideoCapture(self.camera, self.backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.camera}. "
                "Close other camera apps or try another --camera index."
            )

        configure_capture(
            self.cv2,
            capture,
            self.system,
            self.width,
            self.height,
            self.fps,
        )
        self._capture = capture

        ok, frame = capture.read()
        if not ok:
            capture.release()
            self._capture = None
            raise RuntimeError(f"Camera {self.camera} opened but returned no frame.")
        try:
            self._publish(frame)
        except Exception:
            capture.release()
            self._capture = None
            raise

        actual_width = round(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = round(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(self.cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_name(capture.get(self.cv2.CAP_PROP_FOURCC))
        print(
            f"Camera {self.camera}: backend={capture.getBackendName()}, "
            f"format={actual_fourcc}, "
            f"{actual_width}x{actual_height} @ {actual_fps:g} FPS"
        )
        if self.system == "linux" and actual_fourcc not in ("MJPG", "JPEG"):
            print(
                "Warning: the camera did not accept MJPEG; "
                "the stream may have a lower frame rate.",
                file=sys.stderr,
            )

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="webcam-capture",
            daemon=True,
        )
        self._thread.start()

    def _publish(self, frame) -> None:
        ok, encoded = self.cv2.imencode(
            ".jpg",
            frame,
            [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode a camera frame as JPEG.")
        with self._condition:
            self._frame = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                assert self._capture is not None
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames.")
                self._publish(frame)
        except Exception as exc:
            with self._condition:
                self._error = str(exc)
                self._condition.notify_all()
        finally:
            if self._capture is not None:
                self._capture.release()

    def wait_for_frame(self, after: int, timeout: float = 5.0) -> tuple[int, bytes]:
        """Wait for and return a frame newer than *after*."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._sequence > after
                    or self._error is not None
                    or self._stopping.is_set()
                ),
                timeout,
            )
            if self._error:
                raise RuntimeError(self._error)
            if self._stopping.is_set():
                raise RuntimeError("The camera stream has stopped.")
            if self._frame is None:
                raise RuntimeError("No camera frame is available.")
            if not ready and self._sequence <= after:
                raise TimeoutError("Timed out waiting for the next camera frame.")
            return self._sequence, self._frame

    def snapshot(self) -> bytes:
        """Return the latest JPEG frame without its freshness metadata."""

        _, frame = self.snapshot_with_sequence()
        return frame

    def snapshot_with_sequence(self) -> tuple[int, bytes]:
        """Return the latest frame sequence and JPEG as one locked snapshot."""

        with self._condition:
            if self._error:
                raise RuntimeError(self._error)
            if self._frame is None:
                raise RuntimeError("No camera frame is available.")
            return self._sequence, self._frame

    def health(self) -> tuple[dict[str, object], bool]:
        with self._condition:
            healthy = self._frame is not None and self._error is None
            return {
                "status": "ok" if healthy else "error",
                "camera": self.camera,
                "frames_captured": self._sequence,
                "error": self._error,
            }, healthy

    def stop(self) -> None:
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()


class WebcamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        camera: CameraStream,
        display_store: ControllerDisplayStore | None = None,
        *,
        task_store: ControllerDisplayStore | None = None,
    ) -> None:
        if display_store is not None and task_store is not None:
            raise ValueError("provide display_store or task_store, not both")
        super().__init__(address, WebcamHandler)
        self.camera = camera
        self.display_store = display_store or task_store or ControllerDisplayStore()
        # Compatibility for prototype callers; both names reference one store.
        self.task_store = self.display_store


class WebcamHandler(BaseHTTPRequestHandler):
    server: WebcamServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/snapshot.jpg":
            try:
                snapshot_with_sequence = getattr(
                    self.server.camera,
                    "snapshot_with_sequence",
                    None,
                )
                if snapshot_with_sequence is None:
                    sequence, frame = 0, self.server.camera.snapshot()
                else:
                    sequence, frame = snapshot_with_sequence()
                self._send(
                    frame,
                    "image/jpeg",
                    headers={FRAME_SEQUENCE_HEADER: str(sequence)},
                )
            except RuntimeError as exc:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        elif path == "/healthz":
            payload, healthy = self.server.camera.health()
            self._send_json(
                payload,
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
            )
        elif path == TASK_API_PATH:
            self._send_json(self.server.display_store.snapshot())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:
        path = urlsplit(self.path).path
        if path != TASK_API_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not is_loopback_address(self.client_address[0]):
            self.close_connection = True
            self._send_json(
                {"error": "task updates are only accepted from loopback clients"},
                HTTPStatus.FORBIDDEN,
            )
            return
        if self.headers.get_content_type() != "application/json":
            self.close_connection = True
            self._send_json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self.close_connection = True
            self._send_json(
                {"error": "Content-Length is required"},
                HTTPStatus.LENGTH_REQUIRED,
            )
            return
        try:
            length = int(content_length)
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._send_json(
                {"error": "Content-Length must be a non-negative integer"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        if length > MAX_TASK_BODY_BYTES:
            self.close_connection = True
            self._send_json(
                {"error": f"request body exceeds {MAX_TASK_BODY_BYTES} bytes"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json(
                {"error": "request body ended before Content-Length bytes arrived"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            value = json.loads(body.decode("utf-8"))
            envelope = self.server.display_store.publish(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                {"error": f"invalid JSON: {exc}"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        except DisplayConflictError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(envelope)

    def do_DELETE(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path != TASK_API_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not is_loopback_address(self.client_address[0]):
            self.close_connection = True
            self._send_json(
                {"error": "task updates are only accepted from loopback clients"},
                HTTPStatus.FORBIDDEN,
            )
            return
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) - {"session_id"} or len(query.get("session_id", ())) > 1:
            self._send_json(
                {"error": "DELETE accepts only one optional session_id"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        session_id = query.get("session_id", [None])[0]
        try:
            envelope = self.server.display_store.reset(session_id=session_id)
        except DisplayConflictError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(envelope)

    def _send_json(
        self,
        payload: object,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        sequence = -1
        try:
            while True:
                sequence, frame = self.server.camera.wait_for_frame(sequence)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (RuntimeError, TimeoutError):
            self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}", file=sys.stderr)


def list_cameras(cv2: ModuleType, backend: int, maximum: int) -> int:
    """Probe camera indexes and print those that return a frame."""
    found: list[int] = []
    print(f"Checking camera indexes 0 through {maximum - 1}...")
    for index in range(maximum):
        capture = cv2.VideoCapture(index, backend)
        try:
            ok, _ = capture.read() if capture.isOpened() else (False, None)
            if ok:
                found.append(index)
                print(f"  camera {index}: available")
        finally:
            capture.release()
    if not found:
        print("No available camera was found.", file=sys.stderr)
        return 1
    return 0


def local_ip() -> str | None:
    """Best-effort LAN address lookup without sending any data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a local webcam to a browser using MJPEG over HTTP."
    )
    parser.add_argument("--camera", type=parse_camera, default=0, help="camera index")
    parser.add_argument(
        "--system",
        choices=SYSTEM_CHOICES,
        default=default_system(),
        help="camera system (Linux uses V4L2 with MJPEG; default: detected)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf"),
        default="auto",
        help="Windows capture backend (try dshow if auto fails)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument(
        "--port", type=bounded_integer(1, 65535), default=1234, help="HTTP port"
    )
    parser.add_argument(
        "--width", type=bounded_integer(160, 7680), default=1280, help="frame width"
    )
    parser.add_argument(
        "--height", type=bounded_integer(120, 4320), default=720, help="frame height"
    )
    parser.add_argument(
        "--fps", type=bounded_integer(1, 240), default=30, help="requested camera FPS"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_integer(1, 100),
        default=80,
        help="JPEG quality",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="probe camera indexes and exit",
    )
    parser.add_argument(
        "--probe-count",
        type=bounded_integer(1, 20),
        default=5,
        help="number of indexes to check with --list-cameras",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.system == "linux" and args.backend != "auto":
        fail("--backend dshow/msmf can only be used with --system windows.")

    cv2 = load_opencv()
    backend = backend_id(cv2, args.system, args.backend)

    if args.list_cameras:
        raise SystemExit(list_cameras(cv2, backend, args.probe_count))

    camera = CameraStream(
        cv2=cv2,
        camera=args.camera,
        system=args.system,
        backend=backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
    )
    try:
        camera.start()
    except RuntimeError as exc:
        fail(str(exc))

    try:
        server = WebcamServer((args.host, args.port), camera)
    except OSError as exc:
        camera.stop()
        fail(f"Could not listen on {args.host}:{args.port}: {exc}")

    print(f"Webcam stream: http://127.0.0.1:{args.port}")
    print(f"Task API:      PUT {TASK_API_PATH} (loopback clients only)")
    if args.host in ("0.0.0.0", "::"):
        address = local_ip()
        if address:
            print(f"LAN stream:    http://{address}:{args.port}")
        print("Warning: LAN mode has no authentication or encryption.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.server_close()
        camera.stop()


if __name__ == "__main__":
    main()
