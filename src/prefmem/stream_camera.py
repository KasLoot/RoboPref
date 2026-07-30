"""Stream a local webcam to a web browser as an MJPEG feed."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import NoReturn
from urllib.parse import urlsplit


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
  </script>
</body>
</html>
"""

SYSTEM_CHOICES = ("windows", "linux")


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
        with self._condition:
            if self._error:
                raise RuntimeError(self._error)
            if self._frame is None:
                raise RuntimeError("No camera frame is available.")
            return self._frame

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

    def __init__(self, address: tuple[str, int], camera: CameraStream) -> None:
        super().__init__(address, WebcamHandler)
        self.camera = camera


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
                self._send(self.server.camera.snapshot(), "image/jpeg")
            except RuntimeError as exc:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        elif path == "/healthz":
            payload, healthy = self.server.camera.health()
            body = json.dumps(payload).encode()
            self._send(
                body,
                "application/json",
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
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
