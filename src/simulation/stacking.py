"""Interactive MuJoCo block-stacking environment with synchronized cameras."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Final

import mujoco
import numpy as np

from prefmem.execution.frames import CameraCalibration, RGBDFrame
from prefmem.task_publisher import ControllerDisplay, DisplayEnvelope


HOME_QPOS: Final[np.ndarray] = np.array(
    [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04],
    dtype=np.float64,
)
TASK_BLOCKS: Final[tuple[str, ...]] = ("red_block", "green_block", "blue_block")
STORED_BLOCKS: Final[tuple[str, ...]] = (
    "yellow_block",
    "purple_block",
    "orange_block",
)
ALL_BLOCKS: Final[tuple[str, ...]] = TASK_BLOCKS + STORED_BLOCKS
TB6C_V00_POSITIONS: Final[
    tuple[tuple[str, tuple[float, float, float]], ...]
] = (
    ("red_block", (0.32, -0.34, 0.026)),
    ("green_block", (0.32, 0.00, 0.026)),
    ("blue_block", (0.32, 0.34, 0.026)),
    ("yellow_block", (0.745, -0.34, 0.026)),
    ("purple_block", (0.76, 0.00, 0.026)),
    ("orange_block", (0.745, 0.34, 0.026)),
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    simulation_time: float
    qpos: np.ndarray
    control: np.ndarray
    object_positions: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class _RenderStateSnapshot:
    """One immutable MuJoCo state handed from physics to camera rendering."""

    state_revision: int
    submission_id: int
    observed_at: float
    simulation_time: float
    data: mujoco.MjData


@dataclass(frozen=True, slots=True)
class _RenderedStream:
    """Rendered pixels and calibration awaiting atomic pair publication."""

    rgb: np.ndarray
    depth_m: np.ndarray
    calibration: CameraCalibration


class InMemoryTaskPublisher:
    """Single display slot for simulation runs that have no camera HTTP page."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision = 0
        self._display: ControllerDisplay | None = None

    @property
    def display(self) -> ControllerDisplay | None:
        with self._lock:
            return self._display

    def publish(self, display: ControllerDisplay) -> DisplayEnvelope:
        if not isinstance(display, ControllerDisplay):
            raise TypeError("display must be a ControllerDisplay")
        with self._lock:
            self._revision += 1
            self._display = display
            return DisplayEnvelope(self._revision, display)

    def reset(self, *, session_id: str | None = None) -> DisplayEnvelope:
        with self._lock:
            if (
                session_id is not None
                and self._display is not None
                and self._display.session_id != session_id
            ):
                return DisplayEnvelope(self._revision, self._display)
            self._revision += 1
            self._display = None
            return DisplayEnvelope(self._revision, None)


class StackingEnvironment:
    """Own all mutable MuJoCo state on one real-time simulation thread."""

    def __init__(
        self,
        *,
        seed: int = 0,
        width: int = 640,
        height: int = 640,
        render_hz: float = 5.0,
        realtime: bool = True,
        viewer: bool = True,
        viewer_camera: str = "overview",
        scene_path: str | Path | None = None,
        start: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        if render_hz <= 0:
            raise ValueError("render_hz must be positive")
        if viewer_camera not in {"sam", "prefmem", "task", "overview"}:
            raise ValueError(
                "viewer_camera must be 'sam', 'prefmem', 'task', or 'overview'"
            )
        source = (
            Path(scene_path)
            if scene_path is not None
            else Path(__file__).resolve().with_name("stacking_scene.xml")
        )
        self.model = mujoco.MjModel.from_xml_path(str(source))
        framebuffer_width = int(self.model.vis.global_.offwidth)
        framebuffer_height = int(self.model.vis.global_.offheight)
        if width > framebuffer_width or height > framebuffer_height:
            raise ValueError(
                "render dimensions "
                f"{width}x{height} exceed the scene's offscreen framebuffer "
                f"{framebuffer_width}x{framebuffer_height}"
            )
        self.data = mujoco.MjData(self.model)
        self.width = int(width)
        self.height = int(height)
        self.render_hz = float(render_hz)
        self.realtime = bool(realtime)
        self.viewer_enabled = bool(viewer)
        self.viewer_camera = viewer_camera
        self._reset_profile = self._detect_reset_profile()
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._render_condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None
        self._pending_render_snapshot: _RenderStateSnapshot | None = None
        self._render_in_flight = False
        # Revisions are reset epochs, not physics-step counters. A render
        # copied before reset is discarded even if it finishes afterward.
        self._state_revision = 0
        self._render_submission_id = 0
        self._last_published_render_submission = 0
        self._last_published_state_revision = 0
        self._periodic_rendering_paused = False
        self._latest_sam_frame: RGBDFrame | None = None
        self._latest_prefmem_frame: RGBDFrame | None = None
        self._sam_frame_sequence = 0
        self._prefmem_frame_sequence = 0
        self._control_target = np.zeros(self.model.nu, dtype=np.float64)
        self._sam_camera_id = self._resolve_camera_id(
            "sam_camera",
            fallback="task_camera",
            stream_name="SAM",
        )
        self._prefmem_camera_id = self._resolve_camera_id(
            "prefmem_camera",
            fallback="task_camera",
            stream_name="PrefMem",
        )
        # Compatibility for integrations that inspect the former single task
        # camera directly. The old ``task`` viewer name now means PrefMem's
        # model-facing view; use ``sam`` to inspect detector input explicitly.
        self._camera_id = self._prefmem_camera_id

        self._sam_scene_option = mujoco.MjvOption()
        # Panda visual and collision geometry use groups 2 and 3 respectively.
        # Hiding both only in the SAM stream gives the detector an unoccluded
        # top-down workspace without changing collision/physics state. Sites
        # are an independent visualization category and must also be disabled.
        self._sam_scene_option.geomgroup[2] = 0
        self._sam_scene_option.geomgroup[3] = 0
        self._sam_scene_option.sitegroup[:] = 0

        self._prefmem_scene_option = mujoco.MjvOption()
        # PrefMem must see the physical embodiment. Collision geometry remains
        # hidden because it is diagnostic rendering rather than scene evidence.
        self._prefmem_scene_option.geomgroup[2] = 1
        self._prefmem_scene_option.geomgroup[3] = 0
        self._task_scene_option = self._sam_scene_option
        self._arm_qpos_addresses = self._joint_qpos_addresses(
            tuple(f"joint{index}" for index in range(1, 8))
        )
        self._arm_dof_addresses = self._joint_dof_addresses(
            tuple(f"joint{index}" for index in range(1, 8))
        )
        self._finger_qpos_addresses = self._joint_qpos_addresses(
            ("finger_joint1", "finger_joint2")
        )
        self._reset_locked()
        if start:
            self.start()

    @property
    def arm_qpos_addresses(self) -> tuple[int, ...]:
        return self._arm_qpos_addresses

    @property
    def finger_qpos_addresses(self) -> tuple[int, ...]:
        return self._finger_qpos_addresses

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def reset_profile(self) -> str:
        """Name the deterministic scene-reset contract selected at load time."""

        return self._reset_profile

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread_error = None
        with self._render_condition:
            self._pending_render_snapshot = None
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="robopref-mujoco-render",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="robopref-mujoco-physics",
            daemon=True,
        )
        self._render_thread.start()
        self._thread.start()
        try:
            self.wait_for_sam_frame(timeout=10.0)
            self.wait_for_prefmem_frame(timeout=10.0)
        except BaseException:
            self.close()
            raise

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._render_condition:
            self._pending_render_snapshot = None
            self._render_condition.notify_all()
        with self._frame_condition:
            self._frame_condition.notify_all()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in (self._thread, self._render_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def reset(self) -> None:
        render_snapshot: _RenderStateSnapshot | None = None
        with self._lock:
            self._reset_locked()
            if self._render_thread is not None and self._render_thread.is_alive():
                render_snapshot = self._capture_render_snapshot_locked()
            self._frame_condition.notify_all()
        if render_snapshot is not None:
            self._submit_render_snapshot(render_snapshot)

    @property
    def periodic_rendering_paused(self) -> bool:
        with self._lock:
            return self._periodic_rendering_paused

    def pause_periodic_rendering(self, *, timeout: float = 10.0) -> None:
        """Pause scheduled camera pairs without pausing physics.

        Controller-only evidence runs may use this after their frozen initial
        endpoint because neither model-facing stream has a consumer during
        motion. Explicit endpoint requests remain available while paused.
        """

        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with self._lock:
            self._periodic_rendering_paused = True
        deadline = time.monotonic() + timeout
        with self._render_condition:
            self._pending_render_snapshot = None
            while self._render_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out pausing periodic simulation rendering"
                    )
                self._render_condition.wait(remaining)

    def resume_periodic_rendering(self) -> None:
        """Resume scheduled pairs and immediately offer the current state."""

        render_snapshot: _RenderStateSnapshot | None = None
        with self._lock:
            if not self._periodic_rendering_paused:
                return
            self._periodic_rendering_paused = False
            if self._render_thread is not None and self._render_thread.is_alive():
                render_snapshot = self._capture_render_snapshot_locked()
        if render_snapshot is not None:
            self._submit_render_snapshot(render_snapshot)

    def render_camera_pair(
        self,
        *,
        timeout: float = 10.0,
    ) -> tuple[RGBDFrame, RGBDFrame]:
        """Render and wait for one current synchronized SAM/PrefMem pair."""

        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError("the simulation thread failed") from self._thread_error
            if self._render_thread is None or not self._render_thread.is_alive():
                raise RuntimeError("the simulation render worker is not running")
            snapshot = self._capture_render_snapshot_locked()
        self._submit_render_snapshot(snapshot)
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while True:
                if self._thread_error is not None:
                    raise RuntimeError(
                        "the simulation thread failed"
                    ) from self._thread_error
                if snapshot.state_revision != self._state_revision:
                    raise RuntimeError(
                        "the simulator reset during an explicit camera request"
                    )
                if (
                    self._last_published_render_submission
                    >= snapshot.submission_id
                    and self._last_published_state_revision
                    == snapshot.state_revision
                    and self._latest_sam_frame is not None
                    and self._latest_prefmem_frame is not None
                ):
                    return (
                        self._latest_sam_frame,
                        self._latest_prefmem_frame,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for an explicit simulation camera pair"
                    )
                self._frame_condition.wait(remaining)

    def sam_rgbd_frame(self) -> RGBDFrame:
        """Return the newest robot-hidden RGB-D frame supplied only to SAM."""

        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError(
                    "the simulation thread failed"
                ) from self._thread_error
            if self._latest_sam_frame is None:
                raise RuntimeError("the simulator has not rendered a SAM frame yet")
            return self._latest_sam_frame

    def prefmem_rgbd_frame(self) -> RGBDFrame:
        """Return the newest third-person RGB-D frame used by PrefMem."""

        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError(
                    "the simulation thread failed"
                ) from self._thread_error
            if self._latest_prefmem_frame is None:
                raise RuntimeError("the simulator has not rendered a PrefMem frame yet")
            return self._latest_prefmem_frame

    def rgbd_frame(self) -> RGBDFrame:
        """Backward-compatible alias for the executor/SAM RGB-D stream."""

        return self.sam_rgbd_frame()

    def sam_captured_frame(self):
        return self.sam_rgbd_frame().captured_frame()

    def prefmem_captured_frame(self):
        return self.prefmem_rgbd_frame().captured_frame()

    def captured_frame(self):
        """Backward-compatible alias for PrefMem's model-facing stream."""

        return self.prefmem_captured_frame()

    def wait_for_sam_frame(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> RGBDFrame:
        return self._wait_for_stream(
            "SAM",
            lambda: self._latest_sam_frame,
            after_sequence=after_sequence,
            timeout=timeout,
        )

    def wait_for_prefmem_frame(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> RGBDFrame:
        return self._wait_for_stream(
            "PrefMem",
            lambda: self._latest_prefmem_frame,
            after_sequence=after_sequence,
            timeout=timeout,
        )

    def wait_for_frame(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> RGBDFrame:
        """Backward-compatible wait for the executor/SAM RGB-D stream."""

        return self.wait_for_sam_frame(
            after_sequence=after_sequence,
            timeout=timeout,
        )

    def _wait_for_stream(
        self,
        stream_name: str,
        frame_getter: Callable[[], RGBDFrame | None],
        *,
        after_sequence: int,
        timeout: float,
    ) -> RGBDFrame:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            frame = frame_getter()
            while frame is None or frame.sequence <= after_sequence:
                if self._thread_error is not None:
                    raise RuntimeError("the simulation thread failed") from self._thread_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for a {stream_name} simulation frame"
                    )
                self._frame_condition.wait(remaining)
                frame = frame_getter()
            return frame

    def set_control(
        self,
        *,
        arm: np.ndarray | None = None,
        gripper: float | None = None,
    ) -> None:
        with self._lock:
            target = self._control_target.copy()
            if arm is not None:
                values = np.asarray(arm, dtype=np.float64)
                if values.shape != (7,) or not np.all(np.isfinite(values)):
                    raise ValueError("arm control must contain seven finite values")
                target[:7] = values
            if gripper is not None:
                value = float(gripper)
                if not np.isfinite(value) or not 0 <= value <= 0.04:
                    raise ValueError("gripper control must be between 0 and 0.04")
                target[7] = value
            target = np.clip(
                target,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            self._control_target = target

    def arm_qpos(self) -> np.ndarray:
        with self._lock:
            return self.data.qpos[list(self._arm_qpos_addresses)].copy()

    def arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one synchronized arm position/velocity observation."""

        with self._lock:
            positions = self.data.qpos[
                list(self._arm_qpos_addresses)
            ].copy()
            velocities = self.data.qvel[
                list(self._arm_dof_addresses)
            ].copy()
            return positions, velocities

    def gripper_qpos(self) -> float:
        with self._lock:
            values = self.data.qpos[list(self._finger_qpos_addresses)]
            return float(np.mean(values))

    def snapshot(self) -> SimulationSnapshot:
        with self._lock:
            positions = {
                name: self.data.xpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                ].copy()
                for name in ALL_BLOCKS
            }
            qpos = self.data.qpos.copy()
            control = self.data.ctrl.copy()
            qpos.setflags(write=False)
            control.setflags(write=False)
            for position in positions.values():
                position.setflags(write=False)
            return SimulationSnapshot(
                simulation_time=float(self.data.time),
                qpos=qpos,
                control=control,
                object_positions=positions,
            )

    def place_item(
        self,
        name: str,
        position: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Place a free block; useful for scripted disturbances and demos."""

        if name not in ALL_BLOCKS:
            raise ValueError(f"unknown block: {name}")
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position must contain three finite coordinates")
        with self._lock:
            address = self._freejoint_qpos_address(name)
            self.data.qpos[address : address + 3] = point
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
            velocity_address = self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free"
                )
            ]
            self.data.qvel[velocity_address : velocity_address + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _resolve_camera_id(
        self,
        name: str,
        *,
        fallback: str,
        stream_name: str,
    ) -> int:
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            name,
        )
        if camera_id >= 0:
            return int(camera_id)
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            fallback,
        )
        if camera_id < 0:
            raise RuntimeError(
                f"stacking scene has no {name!r} camera for the {stream_name} "
                f"stream and no legacy {fallback!r} fallback"
            )
        return int(camera_id)

    def _detect_reset_profile(self) -> str:
        board_ids = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in ("white_board", "cyan_board")
        )
        if all(object_id >= 0 for object_id in board_ids):
            return "TB6C_V00"
        return "LEGACY_RANDOM"

    def _joint_qpos_addresses(self, names: tuple[str, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for name in names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise RuntimeError(f"scene is missing joint {name}")
            result.append(int(self.model.jnt_qposadr[joint_id]))
        return tuple(result)

    def _joint_dof_addresses(self, names: tuple[str, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for name in names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise RuntimeError(f"scene is missing joint {name}")
            result.append(int(self.model.jnt_dofadr[joint_id]))
        return tuple(result)

    def _freejoint_qpos_address(self, body_name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free"
        )
        if joint_id < 0:
            raise RuntimeError(f"scene is missing free joint for {body_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _reset_locked(self) -> None:
        self._state_revision += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[list(self._arm_qpos_addresses)] = HOME_QPOS[:7]
        self.data.qpos[list(self._finger_qpos_addresses)] = HOME_QPOS[7:]
        self._control_target[:7] = HOME_QPOS[:7]
        self._control_target[7] = 0.04
        self.data.ctrl[:] = self._control_target
        if self._reset_profile == "TB6C_V00":
            block_positions = TB6C_V00_POSITIONS
        else:
            task_positions = self._sample_task_positions()
            block_positions = tuple(
                (name, (*point, 0.026))
                for name, point in zip(TASK_BLOCKS, task_positions, strict=True)
            ) + tuple(
                (name, (*point, 0.056))
                for name, point in zip(
                    STORED_BLOCKS,
                    ((-1.30, -0.72), (-1.15, -0.72), (-1.00, -0.72)),
                    strict=True,
                )
            )
        for name, point in block_positions:
            address = self._freejoint_qpos_address(name)
            self.data.qpos[address : address + 3] = point
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._latest_sam_frame = None
        self._latest_prefmem_frame = None

    def _sample_task_positions(self) -> tuple[tuple[float, float], ...]:
        positions: list[tuple[float, float]] = []
        for _ in TASK_BLOCKS:
            for _attempt in range(500):
                point = (
                    float(self._rng.uniform(0.30, 0.74)),
                    float(self._rng.uniform(-0.30, 0.30)),
                )
                # Keep the mat centre free and prevent initial overlaps.
                outside_mat = not (
                    0.37 <= point[0] <= 0.73
                    and -0.21 <= point[1] <= 0.21
                )
                separated = all(
                    np.hypot(point[0] - old[0], point[1] - old[1]) >= 0.09
                    for old in positions
                )
                if outside_mat and separated:
                    positions.append(point)
                    break
            else:
                raise RuntimeError("could not sample non-overlapping block positions")
        return tuple(positions)

    def _run(self) -> None:
        viewer = None
        try:
            if self.viewer_enabled:
                try:
                    from mujoco import viewer as mj_viewer

                    viewer = mj_viewer.launch_passive(
                        self.model,
                        self.data,
                        show_left_ui=True,
                        show_right_ui=True,
                    )
                    self._configure_viewer(viewer)
                except Exception as error:
                    LOGGER.warning(
                        "Could not open the interactive MuJoCo viewer: %s",
                        error,
                    )
                    viewer = None
            render_period = 1.0 / self.render_hz
            next_render = 0.0
            schedule_revision = -1
            wall_start = time.monotonic()
            while not self._stop_event.is_set():
                render_snapshot: _RenderStateSnapshot | None = None
                with self._lock:
                    if schedule_revision != self._state_revision:
                        schedule_revision = self._state_revision
                        next_render = 0.0
                        # Reset rewinds MuJoCo time.  Rebase real-time pacing so
                        # the physics thread does not wait for the old epoch.
                        wall_start = time.monotonic() - float(self.data.time)
                    self.data.ctrl[:] = self._control_target
                    mujoco.mj_step(self.model, self.data)
                    if self.data.time >= next_render:
                        if not self._periodic_rendering_paused:
                            render_snapshot = (
                                self._capture_render_snapshot_locked()
                            )
                        next_render = self.data.time + render_period
                    if viewer is not None:
                        if viewer.is_running():
                            viewer.sync()
                        else:
                            viewer.close()
                            viewer = None
                    simulation_time = float(self.data.time)
                if render_snapshot is not None:
                    self._submit_render_snapshot(render_snapshot)
                if self.realtime:
                    delay = wall_start + simulation_time - time.monotonic()
                    if delay > 0:
                        self._stop_event.wait(min(delay, 0.01))
        except BaseException as error:
            self._record_thread_error(error)
        finally:
            if viewer is not None:
                viewer.close()

    def _configure_viewer(self, viewer) -> None:
        """Configure only the interactive view; both model streams are independent."""

        if self.viewer_camera == "sam":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self._sam_camera_id
            self._copy_scene_option(viewer.opt, self._sam_scene_option)
            return
        if self.viewer_camera in {"task", "prefmem"}:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self._prefmem_camera_id
            self._copy_scene_option(viewer.opt, self._prefmem_scene_option)
            return
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = (0.0, -0.15, 0.15)
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -28
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 0

    @staticmethod
    def _copy_scene_option(target, source: mujoco.MjvOption) -> None:
        """Copy supported visualization arrays into a viewer option object."""

        target.geomgroup[:] = source.geomgroup
        for name in ("sitegroup", "jointgroup", "tendongroup", "actuatorgroup", "flags"):
            target_value = getattr(target, name, None)
            source_value = getattr(source, name, None)
            if target_value is not None and source_value is not None:
                target_value[:] = source_value

    def _capture_render_snapshot_locked(self) -> _RenderStateSnapshot:
        """Copy authoritative state quickly while the physics lock is held."""

        render_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(render_data, self.model, self.data)
        self._render_submission_id += 1
        return _RenderStateSnapshot(
            state_revision=self._state_revision,
            submission_id=self._render_submission_id,
            observed_at=time.monotonic(),
            simulation_time=float(self.data.time),
            data=render_data,
        )

    def _submit_render_snapshot(self, snapshot: _RenderStateSnapshot) -> None:
        """Offer the newest scheduled camera state to the render worker.

        Camera consumers need current evidence, not a backlog.  If rendering is
        slower than the requested stream rate, replace only an unconsumed
        snapshot; the worker's in-flight snapshot remains immutable.  Reset
        revisions take precedence, so an old physics iteration cannot replace
        a newer post-reset request after releasing the environment lock.
        """

        with self._render_condition:
            if self._stop_event.is_set():
                return
            pending = self._pending_render_snapshot
            if pending is not None and (
                snapshot.state_revision,
                snapshot.submission_id,
            ) <= (
                pending.state_revision,
                pending.submission_id,
            ):
                return
            self._pending_render_snapshot = snapshot
            self._render_condition.notify_all()

    def _render_loop(self) -> None:
        renderer: mujoco.Renderer | None = None
        try:
            renderer = mujoco.Renderer(
                self.model,
                height=self.height,
                width=self.width,
            )
            while True:
                with self._render_condition:
                    while (
                        self._pending_render_snapshot is None
                        and not self._stop_event.is_set()
                    ):
                        self._render_condition.wait()
                    if self._stop_event.is_set():
                        self._pending_render_snapshot = None
                        return
                    snapshot = self._pending_render_snapshot
                    self._pending_render_snapshot = None
                    self._render_in_flight = True
                assert snapshot is not None
                try:
                    sam_render, prefmem_render = self._render_snapshot_pair(
                        renderer,
                        snapshot,
                    )
                    with self._frame_condition:
                        # Rendering is intentionally outside the physics lock.
                        # A reset may complete while pixels are in flight;
                        # never let that old frame repopulate the cleared
                        # post-reset streams.
                        if snapshot.state_revision != self._state_revision:
                            continue
                        if (
                            snapshot.submission_id
                            <= self._last_published_render_submission
                        ):
                            continue
                        self._sam_frame_sequence += 1
                        self._prefmem_frame_sequence += 1
                        sam_frame = RGBDFrame(
                            rgb=sam_render.rgb,
                            depth_m=sam_render.depth_m,
                            calibration=sam_render.calibration,
                            observed_at=snapshot.observed_at,
                            sequence=self._sam_frame_sequence,
                            simulation_time=snapshot.simulation_time,
                        )
                        prefmem_frame = RGBDFrame(
                            rgb=prefmem_render.rgb,
                            depth_m=prefmem_render.depth_m,
                            calibration=prefmem_render.calibration,
                            observed_at=snapshot.observed_at,
                            sequence=self._prefmem_frame_sequence,
                            simulation_time=snapshot.simulation_time,
                        )
                        # Publish the synchronized pair atomically. Consumers
                        # can never observe cameras from different snapshots.
                        self._latest_sam_frame = sam_frame
                        self._latest_prefmem_frame = prefmem_frame
                        self._last_published_render_submission = (
                            snapshot.submission_id
                        )
                        self._last_published_state_revision = (
                            snapshot.state_revision
                        )
                        self._frame_condition.notify_all()
                finally:
                    with self._render_condition:
                        self._render_in_flight = False
                        self._render_condition.notify_all()
        except BaseException as error:
            self._record_thread_error(error)
        finally:
            with self._render_condition:
                self._render_in_flight = False
                self._render_condition.notify_all()
            if renderer is not None:
                renderer.close()

    def _render_snapshot_pair(
        self,
        renderer: mujoco.Renderer,
        snapshot: _RenderStateSnapshot,
    ) -> tuple[_RenderedStream, _RenderedStream]:
        sam = self._render_rgbd_snapshot(
            renderer,
            snapshot.data,
            camera_id=self._sam_camera_id,
            scene_option=self._sam_scene_option,
        )
        prefmem = self._render_rgbd_snapshot(
            renderer,
            snapshot.data,
            camera_id=self._prefmem_camera_id,
            scene_option=self._prefmem_scene_option,
        )
        return sam, prefmem

    def _render_rgbd_snapshot(
        self,
        renderer: mujoco.Renderer,
        render_data: mujoco.MjData,
        *,
        camera_id: int,
        scene_option: mujoco.MjvOption,
    ) -> _RenderedStream:
        renderer.disable_depth_rendering()
        renderer.update_scene(
            render_data,
            camera=camera_id,
            scene_option=scene_option,
        )
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(
            render_data,
            camera=camera_id,
            scene_option=scene_option,
        )
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()

        rotation = render_data.cam_xmat[camera_id].reshape(3, 3).copy()
        position = render_data.cam_xpos[camera_id].copy()
        calibration = CameraCalibration.from_fovy(
            width=self.width,
            height=self.height,
            fovy_degrees=float(self.model.cam_fovy[camera_id]),
            camera_position=position,
            camera_rotation=rotation,
        )
        return _RenderedStream(
            rgb=rgb,
            depth_m=depth,
            calibration=calibration,
        )

    def _record_thread_error(self, error: BaseException) -> None:
        with self._frame_condition:
            if self._thread_error is None:
                self._thread_error = error
            self._stop_event.set()
            self._frame_condition.notify_all()
        with self._render_condition:
            self._render_condition.notify_all()


__all__ = [
    "ALL_BLOCKS",
    "HOME_QPOS",
    "InMemoryTaskPublisher",
    "SimulationSnapshot",
    "StackingEnvironment",
    "TB6C_V00_POSITIONS",
]
