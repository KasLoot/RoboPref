"""Synchronized publication video generation and integrity auditing.

Two H.264 streams are produced from the same frame clock: a 1920x1080 evidence
composite and a clean robot-camera stream.  Source-frame hashes and elapsed-time
mapping are retained in ``video_metadata.json`` so model request frames, events,
and video can be compared without relying on visual guesswork.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np

from experiments.harness.recording import (
    ClockOrigin,
    RecordingError,
    durable_json,
    redact_text,
    sha256_file,
)


class VideoError(RecordingError):
    """Raised when synchronized evidence video cannot be safely finalized."""


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class VideoConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 10
    robot_width: int = 640
    robot_height: int = 480
    pre_roll_seconds: float = 0.5
    final_hold_seconds: float = 1.0
    crf: int = 18
    preset: str = "veryfast"
    maximum_declared_freeze_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps", "robot_width", "robot_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width != 1920 or self.height != 1080 or self.fps != 10:
            raise ValueError("publication composite must be exactly 1920x1080 at 10 FPS")
        if self.width % 2 or self.height % 2 or self.robot_width % 2 or self.robot_height % 2:
            raise ValueError("H.264 output dimensions must be even")
        for name in (
            "pre_roll_seconds",
            "final_hold_seconds",
            "maximum_declared_freeze_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if not self.preset:
            raise ValueError("preset must be non-empty")


@dataclass(frozen=True, slots=True)
class OverlayState:
    episode: str
    profile: str
    scenario: str
    seed: int
    system_state: str = "INITIALIZING"
    phase: str = "SETUP"
    active_agent: str = "none"
    latest_user_request: str = ""
    latest_system_output: str = ""
    safety_event: str = ""
    monitor_event: str = ""
    validator_event: str = ""
    terminal_lines: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceFrameReference:
    third_person_id: str | None = None
    robot_camera_id: str | None = None
    third_person_sha256: str | None = None
    robot_camera_sha256: str | None = None


def source_frame_sha256(frame: np.ndarray) -> str:
    """Hash exact source pixels before resizing or video compression."""

    image = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _as_bgr(frame: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise VideoError(f"{name} must have uint8 pixels")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise VideoError(f"{name} must be HxW, HxWx3, or HxWx4")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise VideoError(f"{name} is too small")
    return np.ascontiguousarray(image)


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def _clip(text: object, limit: int) -> str:
    value = redact_text(str(text)).replace("\n", " ").replace("\r", " ")
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def _put_line(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def compose_evidence_frame(
    third_person: np.ndarray,
    robot_camera: np.ndarray,
    *,
    overlay: OverlayState,
    utc: str,
    elapsed_seconds: float,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Build a fixed-layout composite whose text never covers the scene panes."""

    if width != 1920 or height != 1080:
        raise ValueError("composite layout is defined only for 1920x1080")
    third = _as_bgr(third_person, name="third_person")
    robot = _as_bgr(robot_camera, name="robot_camera")
    canvas = np.full((height, width, 3), 15, dtype=np.uint8)

    # Scene panes have dedicated pixels: overlays live in the adjacent header,
    # metadata, and terminal regions and therefore cannot obscure observations.
    canvas[40:720, 0:1280] = _letterbox(third, 1280, 680)
    canvas[40:520, 1280:1920] = _letterbox(robot, 640, 480)
    canvas[0:40, :] = (27, 27, 27)
    canvas[520:720, 1280:1920] = (31, 31, 31)
    canvas[720:1080, :] = (20, 20, 20)
    cv2.rectangle(canvas, (0, 40), (1279, 719), (105, 105, 105), 1)
    cv2.rectangle(canvas, (1280, 40), (1919, 519), (105, 105, 105), 1)
    cv2.rectangle(canvas, (1280, 520), (1919, 719), (105, 105, 105), 1)
    cv2.rectangle(canvas, (0, 720), (1919, 1079), (105, 105, 105), 1)

    _put_line(canvas, "THIRD-PERSON SIMULATOR", (12, 27), scale=0.55)
    _put_line(canvas, "EXACT ROBOT CAMERA", (1292, 27), scale=0.55)
    header = (
        f"episode={_clip(overlay.episode, 36)}  profile={_clip(overlay.profile, 24)}  "
        f"scenario={_clip(overlay.scenario, 36)}  seed={overlay.seed}"
    )
    _put_line(canvas, header, (12, 747), scale=0.55, color=(255, 225, 150))
    _put_line(
        canvas,
        f"utc={_clip(utc, 40)}  elapsed={elapsed_seconds:010.3f}s",
        (12, 774),
        scale=0.55,
    )
    _put_line(
        canvas,
        f"state={_clip(overlay.system_state, 35)}  phase={_clip(overlay.phase, 25)}  "
        f"agent={_clip(overlay.active_agent, 30)}",
        (12, 801),
        scale=0.55,
    )
    _put_line(
        canvas,
        f"USER: {_clip(overlay.latest_user_request, 190)}",
        (12, 835),
        scale=0.50,
        color=(180, 230, 255),
    )
    _put_line(
        canvas,
        f"SYSTEM: {_clip(overlay.latest_system_output, 185)}",
        (12, 861),
        scale=0.50,
        color=(190, 255, 190),
    )

    info_lines = (
        ("SYSTEM STATE", overlay.system_state, (255, 225, 150)),
        ("PHASE", overlay.phase, (235, 235, 235)),
        ("ACTIVE AGENT", overlay.active_agent, (235, 235, 235)),
        ("SAFETY", overlay.safety_event or "none", (160, 190, 255)),
        ("MONITOR", overlay.monitor_event or "none", (190, 255, 190)),
        ("VALIDATOR", overlay.validator_event or "none", (255, 210, 170)),
    )
    y = 545
    for label, value, color in info_lines:
        _put_line(canvas, f"{label}: {_clip(value, 58)}", (1292, y), scale=0.48, color=color)
        y += 27

    _put_line(canvas, "TERMINAL / EVENT LOG (REDACTED)", (12, 898), scale=0.50)
    terminal = tuple(overlay.terminal_lines)[-6:]
    for index in range(6):
        text = terminal[index] if index < len(terminal) else ""
        _put_line(canvas, _clip(text, 220), (12, 928 + index * 24), scale=0.44)
    return canvas


class _FFmpegWriter:
    def __init__(
        self,
        output_partial: Path,
        *,
        width: int,
        height: int,
        fps: int,
        crf: int,
        preset: str,
    ) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise VideoError("ffmpeg is required for H.264 evidence recording")
        self.output_partial = output_partial
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-g",
            str(fps * 2),
            "-keyint_min",
            str(fps * 2),
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            "-y",
            str(output_partial),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise VideoError(
                f"encoder frame must be uint8 {self.height}x{self.width}x3"
            )
        if self.process.stdin is None:
            raise VideoError("ffmpeg input is closed")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as error:
            details = self._stderr_excerpt()
            raise VideoError(f"ffmpeg stopped while encoding: {details}") from error
        self.frames += 1

    def _stderr_excerpt(self) -> str:
        if self.process.poll() is None or self.process.stderr is None:
            return "no encoder diagnostic available"
        try:
            return self.process.stderr.read().decode("utf-8", errors="replace")[-1000:]
        except OSError:
            return "could not read encoder diagnostic"

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=60)
        if return_code != 0:
            details = self._stderr_excerpt()
            if self.process.stderr is not None:
                self.process.stderr.close()
            raise VideoError(f"ffmpeg exited with code {return_code}: {details}")
        if self.process.stderr is not None:
            self.process.stderr.close()
        if not self.output_partial.is_file() or self.output_partial.stat().st_size == 0:
            raise VideoError(f"ffmpeg did not create {self.output_partial.name}")

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stderr is not None:
            self.process.stderr.close()


class SynchronizedVideoRecorder:
    """Encode composite and robot-camera frames on one constant-rate timeline."""

    def __init__(
        self,
        attempt_dir: Path,
        *,
        origin: ClockOrigin,
        config: VideoConfig | None = None,
        secret_values: Iterable[str] = (),
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.origin = origin
        self.config = config or VideoConfig()
        self.secret_values = tuple(secret_values)
        self.composite_final = self.attempt_dir / "video.mp4"
        self.robot_final = self.attempt_dir / "robot_camera.mp4"
        self.composite_partial = self.attempt_dir / "video.mp4.partial"
        self.robot_partial = self.attempt_dir / "robot_camera.mp4.partial"
        for path in (
            self.composite_final,
            self.robot_final,
            self.composite_partial,
            self.robot_partial,
        ):
            if path.exists():
                raise VideoError(f"refusing to overwrite recording artifact: {path}")
        self._composite = _FFmpegWriter(
            self.composite_partial,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            crf=self.config.crf,
            preset=self.config.preset,
        )
        try:
            self._robot = _FFmpegWriter(
                self.robot_partial,
                width=self.config.robot_width,
                height=self.config.robot_height,
                fps=self.config.fps,
                crf=self.config.crf,
                preset=self.config.preset,
            )
        except Exception:
            self._composite.abort()
            raise
        self._lock = threading.Lock()
        self._started = False
        self._finalized = False
        self._last: tuple[
            np.ndarray,
            np.ndarray,
            OverlayState,
            SourceFrameReference,
            float,
            float,
        ] | None = None
        self._links: list[dict[str, Any]] = []
        self._start_source_elapsed: float | None = None
        self._end_source_elapsed: float | None = None

    @property
    def frame_count(self) -> int:
        return self._composite.frames

    def __enter__(self) -> "SynchronizedVideoRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc is not None and not self._finalized:
            self.abort(reason=f"uncaught {type(exc).__name__}: {exc}")
        return False

    def _redacted_overlay(self, overlay: OverlayState) -> OverlayState:
        values = asdict(overlay)
        for field in (
            "episode",
            "profile",
            "scenario",
            "system_state",
            "phase",
            "active_agent",
            "latest_user_request",
            "latest_system_output",
            "safety_event",
            "monitor_event",
            "validator_event",
        ):
            values[field] = redact_text(str(values[field]), secret_values=self.secret_values)
        values["terminal_lines"] = tuple(
            redact_text(str(item), secret_values=self.secret_values)
            for item in values["terminal_lines"]
        )
        return OverlayState(**values)

    def _emit(
        self,
        third: np.ndarray,
        robot: np.ndarray,
        overlay: OverlayState,
        refs: SourceFrameReference,
        source_elapsed: float,
        *,
        source_capture_elapsed: float | None = None,
        hold_kind: str | None = None,
    ) -> None:
        frame_index = self._composite.frames
        video_elapsed = frame_index / self.config.fps
        instant = self.origin.utc.timestamp() + source_elapsed
        # datetime.fromtimestamp would add an unnecessary local-time dependency.
        from datetime import datetime, timezone

        utc = (
            datetime.fromtimestamp(instant, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        composite = compose_evidence_frame(
            third,
            robot,
            overlay=overlay,
            utc=utc,
            elapsed_seconds=source_elapsed,
            width=self.config.width,
            height=self.config.height,
        )
        clean_robot = _letterbox(
            robot, self.config.robot_width, self.config.robot_height
        )
        self._composite.write(composite)
        self._robot.write(clean_robot)
        third_hash = refs.third_person_sha256 or source_frame_sha256(third)
        robot_hash = refs.robot_camera_sha256 or source_frame_sha256(robot)
        self._links.append(
            {
                "frame_index": frame_index,
                "video_elapsed_seconds": video_elapsed,
                "source_elapsed_seconds": source_elapsed,
                "source_capture_elapsed_seconds": (
                    source_elapsed
                    if source_capture_elapsed is None
                    else source_capture_elapsed
                ),
                "third_person_id": refs.third_person_id,
                "third_person_sha256": third_hash,
                "robot_camera_id": refs.robot_camera_id,
                "robot_camera_sha256": robot_hash,
                "hold_kind": hold_kind,
            }
        )

    def write(
        self,
        third_person: np.ndarray,
        robot_camera: np.ndarray,
        *,
        overlay: OverlayState,
        source_elapsed_seconds: float | None = None,
        source_reference: SourceFrameReference | None = None,
    ) -> int:
        """Write one source tick; the first tick is preceded by declared pre-roll."""

        with self._lock:
            if self._finalized:
                raise VideoError("video recorder has been finalized")
            clock_sampled_here = source_elapsed_seconds is None
            elapsed = (
                (self.origin.stamp()["elapsed_seconds"])
                if clock_sampled_here
                else float(source_elapsed_seconds)
            )
            # Preserve acquisition time even if a live caller arrives within
            # the current CFR slot and the encoder must place it in the next
            # representable slot.  Request/frame audits align to acquisition;
            # video duration and overlays align to the emitted CFR timeline.
            captured_elapsed = elapsed
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("source_elapsed_seconds must be finite and non-negative")
            if self._end_source_elapsed is not None and elapsed < self._end_source_elapsed:
                raise VideoError("source frame timestamps must be monotonic")
            if self._started and self._start_source_elapsed is not None:
                # A CFR encoder cannot represent two distinct source/overlay
                # ticks in the same 100 ms slot.  Calls that use our shared
                # live clock wait for the next slot; callers supplying an
                # external simulator timestamp must already be rate-limited.
                next_slot = (
                    self._start_source_elapsed
                    + self.frame_count / self.config.fps
                )
                if elapsed + 1e-9 < next_slot:
                    if not clock_sampled_here:
                        raise VideoError(
                            "source ticks exceed the configured video frame rate"
                        )
                    time.sleep(next_slot - elapsed)
                    elapsed = max(
                        next_slot,
                        float(self.origin.stamp()["elapsed_seconds"]),
                    )
            third = _as_bgr(third_person, name="third_person").copy()
            robot = _as_bgr(robot_camera, name="robot_camera").copy()
            clean_overlay = self._redacted_overlay(overlay)
            refs = source_reference or SourceFrameReference()
            actual_third_hash = source_frame_sha256(third)
            actual_robot_hash = source_frame_sha256(robot)
            if (
                refs.third_person_sha256 is not None
                and refs.third_person_sha256 != actual_third_hash
            ):
                raise VideoError("declared third-person source hash is not exact")
            if (
                refs.robot_camera_sha256 is not None
                and refs.robot_camera_sha256 != actual_robot_hash
            ):
                raise VideoError("declared robot-camera source hash is not exact")
            refs = SourceFrameReference(
                third_person_id=refs.third_person_id,
                robot_camera_id=refs.robot_camera_id,
                third_person_sha256=actual_third_hash,
                robot_camera_sha256=actual_robot_hash,
            )
            if not self._started:
                # A monotonic origin cannot contain footage from negative time.
                # If the first source frame arrives earlier than the configured
                # pre-roll, emit only the elapsed portion instead of compressing
                # a full pre-roll into a shorter source interval.
                start_elapsed = max(0.0, elapsed - self.config.pre_roll_seconds)
                pre_frames = round(
                    (elapsed - start_elapsed) * self.config.fps
                )
                for offset in range(pre_frames):
                    held_elapsed = start_elapsed + offset / self.config.fps
                    self._emit(
                        third,
                        robot,
                        clean_overlay,
                        refs,
                        held_elapsed,
                        source_capture_elapsed=captured_elapsed,
                        hold_kind="pre_roll",
                    )
                self._started = True
                self._start_source_elapsed = start_elapsed
            elif self._last is not None and self._start_source_elapsed is not None:
                # Maintain a true 10 FPS timeline even when model/service work
                # delays the next rendered source frame.  Gaps retain the last
                # synchronized views and are explicitly labelled in metadata.
                (
                    last_third,
                    last_robot,
                    last_overlay,
                    last_refs,
                    _,
                    last_capture_elapsed,
                ) = self._last
                desired_index = round(
                    (elapsed - self._start_source_elapsed) * self.config.fps
                )
                while self.frame_count < desired_index:
                    held_elapsed = (
                        self._start_source_elapsed
                        + self.frame_count / self.config.fps
                    )
                    self._emit(
                        last_third,
                        last_robot,
                        last_overlay,
                        last_refs,
                        held_elapsed,
                        source_capture_elapsed=last_capture_elapsed,
                        hold_kind="source_gap",
                    )
            self._emit(
                third,
                robot,
                clean_overlay,
                refs,
                elapsed,
                source_capture_elapsed=captured_elapsed,
            )
            self._last = (
                third,
                robot,
                clean_overlay,
                refs,
                elapsed,
                captured_elapsed,
            )
            self._end_source_elapsed = elapsed
            return self.frame_count - 1

    def finalize(self, *, final_hold_seconds: float | None = None) -> dict[str, Any]:
        with self._lock:
            if self._finalized:
                raise VideoError("video recorder has already been finalized")
            if self._last is None:
                self._composite.abort()
                self._robot.abort()
                self._finalized = True
                raise VideoError("cannot finalize a recording with no source frames")
            hold = (
                self.config.final_hold_seconds
                if final_hold_seconds is None
                else float(final_hold_seconds)
            )
            if not math.isfinite(hold) or hold < 0:
                raise ValueError("final_hold_seconds must be finite and non-negative")
            third, robot, overlay, refs, last_elapsed, last_capture_elapsed = self._last
            hold_frames = round(hold * self.config.fps)
            for offset in range(1, hold_frames + 1):
                self._emit(
                    third,
                    robot,
                    overlay,
                    refs,
                    last_elapsed + offset / self.config.fps,
                    source_capture_elapsed=last_capture_elapsed,
                    hold_kind="final_hold",
                )
            self._end_source_elapsed = last_elapsed + hold_frames / self.config.fps
            try:
                self._composite.close()
                self._robot.close()
                _fsync_file(self.composite_partial)
                _fsync_file(self.robot_partial)
                os.replace(self.composite_partial, self.composite_final)
                os.replace(self.robot_partial, self.robot_final)
                _fsync_directory(self.attempt_dir)
            except Exception:
                self._composite.abort()
                self._robot.abort()
                # .partial files deliberately remain as crash evidence.
                self._finalized = True
                raise
            metadata = {
                "schema_version": 1,
                "clock_origin": self.origin.to_json(),
                "timeline": {
                    "source_start_elapsed_seconds": self._start_source_elapsed,
                    "source_end_elapsed_seconds": self._end_source_elapsed,
                    "video_start_seconds": 0.0,
                    "video_end_seconds": self.frame_count / self.config.fps,
                },
                "composite": {
                    "path": "video.mp4",
                    "codec": "h264",
                    "width": self.config.width,
                    "height": self.config.height,
                    "fps": self.config.fps,
                    "frame_count": self.frame_count,
                    "layout": {
                        "third_person": [0, 40, 1280, 720],
                        "robot_camera": [1280, 40, 1920, 520],
                        "state_events": [1280, 520, 1920, 720],
                        "terminal_events": [0, 720, 1920, 1080],
                    },
                },
                "robot_camera": {
                    "path": "robot_camera.mp4",
                    "codec": "h264",
                    "width": self.config.robot_width,
                    "height": self.config.robot_height,
                    "fps": self.config.fps,
                    "frame_count": self._robot.frames,
                },
                "recording_policy": {
                    "pre_roll_seconds": self.config.pre_roll_seconds,
                    "final_hold_seconds": hold,
                    "maximum_declared_freeze_seconds": self.config.maximum_declared_freeze_seconds,
                    "constant_frame_rate": True,
                    "scene_panes_obscured_by_overlay": False,
                },
                "overlay_schema": [
                    "episode",
                    "profile",
                    "scenario",
                    "seed",
                    "utc",
                    "elapsed",
                    "system_state",
                    "phase",
                    "active_agent",
                    "latest_user_request",
                    "latest_system_output",
                    "safety_event",
                    "monitor_event",
                    "validator_event",
                    "redacted_terminal_events",
                ],
                "source_frame_linkage": self._links,
            }
            durable_json(
                self.attempt_dir / "video_metadata.json", metadata, exclusive=True
            )
            self._finalized = True
            return metadata

    def abort(self, *, reason: str) -> Path:
        """Stop encoders and retain partial MP4s with an explicit marker."""

        with self._lock:
            if self._finalized:
                raise VideoError("video recorder has already been finalized")
            self._composite.abort()
            self._robot.abort()
            marker = self.attempt_dir / "video_partial.json"
            durable_json(
                marker,
                {
                    "schema_version": 1,
                    "artifact_state": "PARTIAL",
                    "reason": redact_text(reason, secret_values=self.secret_values),
                    "clock_origin": self.origin.to_json(),
                    "frames_written": self.frame_count,
                    "partial_files": [
                        path.name
                        for path in (self.composite_partial, self.robot_partial)
                        if path.exists()
                    ],
                    "source_frame_linkage": self._links,
                },
            )
            self._finalized = True
            return marker


def _probe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise VideoError("ffprobe is required for evidence video audit")
    command = [
        executable,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise VideoError(f"ffprobe failed for {path.name}: {completed.stderr[-1000:]}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise VideoError(f"ffprobe returned no video stream for {path.name}") from error
    return {"stream": stream, "format": payload.get("format", {})}


def _rate(value: object) -> float:
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as error:
        raise VideoError(f"invalid ffprobe frame rate {value!r}") from error
    if not math.isfinite(result):
        raise VideoError(f"non-finite ffprobe frame rate {value!r}")
    return result


def _decode_at(path: Path, seconds: float) -> bool:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise VideoError("ffmpeg is required for frame decodability checks")
    command = [
        executable,
        "-v",
        "error",
        "-ss",
        f"{max(0.0, seconds):.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=60)
    return completed.returncode == 0 and completed.stdout.startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def audit_video_file(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: float = 10.0,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    """Use ffprobe plus first/middle/last decoding to validate one MP4."""

    video = Path(path)
    errors: list[str] = []
    if not video.is_file() or video.is_symlink() or video.stat().st_size == 0:
        return {"passed": False, "path": str(video), "errors": ["missing or empty video"]}
    try:
        probe = _probe(video)
        stream = probe["stream"]
        codec = str(stream.get("codec_name", ""))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps = _rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        frame_text = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
        frame_count = int(frame_text)
        duration = float(stream.get("duration") or probe["format"].get("duration") or 0)
    except (ValueError, TypeError, VideoError) as error:
        return {"passed": False, "path": str(video), "errors": [str(error)]}
    if codec not in {"h264", "avc1"}:
        errors.append(f"expected H.264 codec, got {codec!r}")
    if (width, height) != (expected_width, expected_height):
        errors.append(
            f"expected {expected_width}x{expected_height}, got {width}x{height}"
        )
    if abs(fps - expected_fps) > 1e-6:
        errors.append(f"expected {expected_fps:g} FPS, got {fps:g}")
    if frame_count <= 0 or duration <= 0:
        errors.append("video has no sensible frame count or duration")
    if expected_frames is not None and frame_count != expected_frames:
        errors.append(f"expected {expected_frames} frames, got {frame_count}")
    if frame_count and duration and abs(duration - frame_count / fps) > max(0.15, 1.5 / fps):
        errors.append("duration and frame count disagree")
    sample_times = {
        "first": 0.0,
        "middle": max(0.0, duration / 2),
        "last": max(0.0, duration - 1 / max(fps, 1)),
    }
    decodable = {label: _decode_at(video, value) for label, value in sample_times.items()}
    for label, ok in decodable.items():
        if not ok:
            errors.append(f"{label} frame is not decodable")
    return {
        "passed": not errors,
        "path": str(video),
        "sha256": sha256_file(video),
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "decodable": decodable,
        "errors": errors,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise VideoError(f"{path.name}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _sample_composite_panes(path: Path, frame_count: int) -> list[str]:
    errors: list[str] = []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return ["could not open composite for pane audit"]
    try:
        indexes = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                errors.append(f"could not decode composite pane sample {index}")
                continue
            regions = {
                "third_person": frame[40:720, 0:1280],
                "robot_camera": frame[40:520, 1280:1920],
                "state_events": frame[520:720, 1280:1920],
                "terminal_events": frame[720:1080, 0:1920],
            }
            for name, region in regions.items():
                # A valid pane can be uniform (for example a calibration
                # background), but an all-black encoder placeholder cannot.
                if region.size == 0 or float(np.mean(region)) < 0.75:
                    errors.append(f"{name} pane is blank at frame {index}")
            for name in ("state_events", "terminal_events"):
                # Dedicated text panels always contain required labels.  Their
                # luminance range is a robust, OCR-free visibility check.
                region = regions[name]
                if float(np.max(region) - np.min(region)) < 20:
                    errors.append(f"required overlay is not visible in {name} at frame {index}")
    finally:
        capture.release()
    return errors


def audit_attempt_videos(attempt_dir: Path) -> dict[str, Any]:
    """Audit both streams, overlays, shared timing, and source-frame linkage."""

    root = Path(attempt_dir)
    errors: list[str] = []
    metadata_path = root / "video_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"passed": False, "errors": [f"invalid video_metadata.json: {error}"]}
    try:
        composite_meta = metadata["composite"]
        robot_meta = metadata["robot_camera"]
        if composite_meta.get("path") != "video.mp4":
            raise VideoError("composite path must be video.mp4")
        if robot_meta.get("path") != "robot_camera.mp4":
            raise VideoError("robot camera path must be robot_camera.mp4")
        expected_frames = int(composite_meta["frame_count"])
        composite = audit_video_file(
            root / composite_meta["path"],
            expected_width=1920,
            expected_height=1080,
            expected_fps=10,
            expected_frames=expected_frames,
        )
        robot = audit_video_file(
            root / robot_meta["path"],
            expected_width=int(robot_meta["width"]),
            expected_height=int(robot_meta["height"]),
            expected_fps=10,
            expected_frames=int(robot_meta["frame_count"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError, VideoError) as error:
        return {"passed": False, "errors": [f"invalid video metadata contract: {error}"]}
    errors.extend(f"composite: {item}" for item in composite["errors"])
    errors.extend(f"robot_camera: {item}" for item in robot["errors"])
    if composite.get("frame_count", 0):
        errors.extend(
            _sample_composite_panes(root / "video.mp4", int(composite["frame_count"]))
        )
    if composite.get("frame_count") != robot.get("frame_count"):
        errors.append("composite and robot camera frame counts differ")
    if set(metadata.get("overlay_schema", ())) != {
        "episode",
        "profile",
        "scenario",
        "seed",
        "utc",
        "elapsed",
        "system_state",
        "phase",
        "active_agent",
        "latest_user_request",
        "latest_system_output",
        "safety_event",
        "monitor_event",
        "validator_event",
        "redacted_terminal_events",
    }:
        errors.append("required composite overlay schema is incomplete")
    layout = composite_meta.get("layout", {})
    expected_layout = {
        "third_person": [0, 40, 1280, 720],
        "robot_camera": [1280, 40, 1920, 520],
        "state_events": [1280, 520, 1920, 720],
        "terminal_events": [0, 720, 1920, 1080],
    }
    for pane, expected_region in expected_layout.items():
        if layout.get(pane) != expected_region:
            errors.append(f"composite layout has invalid {pane} pane")

    linkage = metadata.get("source_frame_linkage")
    if not isinstance(linkage, list) or len(linkage) != expected_frames:
        errors.append("source-frame linkage count differs from encoded frame count")
        linkage = []
    else:
        for expected_index, link in enumerate(linkage):
            if link.get("frame_index") != expected_index:
                errors.append(f"source-frame linkage is discontinuous at {expected_index}")
                break
            for field in ("third_person_sha256", "robot_camera_sha256"):
                value = link.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    errors.append(f"source-frame linkage {expected_index} lacks {field}")
                    break

    policy = metadata.get("recording_policy", {})
    freeze_limit = float(policy.get("maximum_declared_freeze_seconds", 0))
    if linkage and freeze_limit >= 0:
        longest = 1
        current = 1
        previous: tuple[str, str] | None = None
        for link in linkage:
            pair = (link["third_person_sha256"], link["robot_camera_sha256"])
            if pair == previous and link.get("hold_kind") not in {
                "pre_roll",
                "final_hold",
            }:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
            previous = pair
        if longest / 10 > freeze_limit:
            errors.append(
                f"undeclared frozen source panes last {longest / 10:.1f}s "
                f"(limit {freeze_limit:.1f}s)"
            )

    timeline = metadata.get("timeline", {})
    try:
        source_start = float(timeline["source_start_elapsed_seconds"])
        source_end = float(timeline["source_end_elapsed_seconds"])
    except (KeyError, TypeError, ValueError):
        errors.append("video timeline lacks source elapsed bounds")
        source_start, source_end = 0.0, -1.0
    if source_end < source_start:
        errors.append("video source elapsed interval is invalid")
    tolerance = 1 / 10 + 1e-6
    if composite.get("duration_seconds") is not None and abs(
        float(composite["duration_seconds"]) - (source_end - source_start)
    ) > 1.5 / 10:
        errors.append("event/source interval and encoded video duration disagree")
    try:
        events = _load_jsonl(root / "events.jsonl")
        for event in events:
            elapsed = event.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)) and not (
                source_start - tolerance <= float(elapsed) <= source_end + tolerance
            ):
                errors.append(
                    f"event {event.get('event_id', '?')} lies outside recording interval"
                )
            video_time = event.get("video_time_seconds")
            if (
                isinstance(elapsed, (int, float))
                and isinstance(video_time, (int, float))
                and abs(float(video_time) - (float(elapsed) - source_start))
                > tolerance
            ):
                errors.append(
                    f"event {event.get('event_id', '?')} video timestamp disagrees with origin mapping"
                )
        frame_requests = _load_jsonl(root / "frame_requests.jsonl")
        for request in frame_requests:
            if request.get("record_type") != "frame_request":
                continue
            elapsed = request.get("capture_elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or not (
                source_start - tolerance <= float(elapsed) <= source_end + tolerance
            ):
                errors.append(
                    f"request frame {request.get('request_id', '?')} lies outside recording interval"
                )
            camera_frame_id = request.get("camera_frame_id")
            if not isinstance(camera_frame_id, str) or not camera_frame_id:
                errors.append(
                    f"request frame {request.get('request_id', '?')} lacks camera_frame_id"
                )
                continue
            requested_source_hash = request.get("source_frame_sha256")
            if (
                not isinstance(requested_source_hash, str)
                or len(requested_source_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in requested_source_hash
                )
            ):
                errors.append(
                    f"request frame {request.get('request_id', '?')} lacks exact source-frame hash"
                )
                continue
            # Most visual calls send the lossless source image directly.  A
            # transport such as SAM may instead send a deterministic JPEG; in
            # that case ``source_image_path`` retains the lossless capture used
            # to derive those exact request bytes.  Never compare a lossy
            # transport image itself to the raw video-source pixel digest.
            image_relative = request.get("source_image_path") or request.get(
                "image_path"
            )
            image_path = (root / str(image_relative)).resolve()
            if not image_path.is_relative_to(root.resolve()):
                errors.append(
                    f"request frame {request.get('request_id', '?')} escapes attempt directory"
                )
                continue
            decoded_request = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded_request is None:
                errors.append(
                    f"request frame {request.get('request_id', '?')} cannot be decoded for source comparison"
                )
                continue
            if source_frame_sha256(decoded_request) != requested_source_hash:
                errors.append(
                    f"request frame {request.get('request_id', '?')} pixels differ from declared video source"
                )
                continue
            linked = [
                item
                for item in linkage
                if item.get("robot_camera_id") == camera_frame_id
                and item.get("robot_camera_sha256") == requested_source_hash
            ]
            if not linked:
                errors.append(
                    f"request frame {request.get('request_id', '?')} has no exact video source linkage"
                )
            elif isinstance(elapsed, (int, float)) and min(
                abs(
                    float(
                        item.get(
                            "source_capture_elapsed_seconds",
                            item["source_elapsed_seconds"],
                        )
                    )
                    - float(elapsed)
                )
                for item in linked
            ) > tolerance:
                errors.append(
                    f"request frame {request.get('request_id', '?')} video linkage is not synchronized"
                )
    except (OSError, UnicodeError, json.JSONDecodeError, VideoError) as error:
        errors.append(f"could not audit event/frame timing: {error}")
    return {
        "schema_version": 1,
        "passed": not errors,
        "composite": composite,
        "robot_camera": robot,
        "source_link_count": len(linkage),
        "errors": errors,
    }


__all__ = [
    "OverlayState",
    "SourceFrameReference",
    "SynchronizedVideoRecorder",
    "VideoConfig",
    "VideoError",
    "audit_attempt_videos",
    "audit_video_file",
    "compose_evidence_frame",
    "source_frame_sha256",
]
