"""Interactive MuJoCo block-stacking environment with synchronized RGB-D."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Final

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
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    simulation_time: float
    qpos: np.ndarray
    control: np.ndarray
    object_positions: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SimulationVideoFrame:
    """One synchronized clean task view and full-scene overview view."""

    sequence: int
    observed_at: float
    simulation_time: float
    task_rgb: np.ndarray
    overview_rgb: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        task = np.asarray(self.task_rgb)
        overview = np.asarray(self.overview_rgb)
        if (
            task.dtype != np.uint8
            or overview.dtype != np.uint8
            or task.ndim != 3
            or overview.shape != task.shape
            or task.shape[2] != 3
        ):
            raise ValueError("video views must be matching HxWx3 uint8 arrays")
        task = task.copy()
        overview = overview.copy()
        task.setflags(write=False)
        overview.setflags(write=False)
        object.__setattr__(self, "task_rgb", task)
        object.__setattr__(self, "overview_rgb", overview)
        object.__setattr__(self, "observed_at", float(self.observed_at))
        object.__setattr__(self, "simulation_time", float(self.simulation_time))


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
        if viewer_camera not in {"task", "overview"}:
            raise ValueError("viewer_camera must be 'task' or 'overview'")
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
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None
        self._latest_frame: RGBDFrame | None = None
        self._latest_video_frame: SimulationVideoFrame | None = None
        self._frame_sequence = 0
        self._control_target = np.zeros(self.model.nu, dtype=np.float64)
        self._camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "task_camera"
        )
        if self._camera_id < 0:
            raise RuntimeError("stacking scene has no task_camera")
        self._task_scene_option = mujoco.MjvOption()
        # Panda visual meshes use geom group 2. Excluding that group from only
        # the fixed task camera creates the requested unoccluded first
        # experiment while the interactive viewer still shows the full robot.
        self._task_scene_option.geomgroup[2] = 0
        self._overview_scene_option = mujoco.MjvOption()
        self._overview_scene_option.geomgroup[2] = 1
        self._overview_camera = mujoco.MjvCamera()
        self._overview_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._overview_camera.fixedcamid = -1
        self._overview_camera.lookat[:] = (0.0, -0.15, 0.15)
        self._overview_camera.distance = 2.5
        self._overview_camera.azimuth = 145
        self._overview_camera.elevation = -28
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

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="robopref-mujoco",
            daemon=True,
        )
        self._thread.start()
        self.wait_for_frame(timeout=10.0)

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))

    def reset(self, *, seed: int | None = None) -> SimulationSnapshot:
        """Reset and return the exact hidden state before the next sim step."""

        with self._lock:
            if seed is not None:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("seed must be an integer or None")
                self.seed = seed
                self._rng = np.random.default_rng(seed)
            self._reset_locked()
            return self.snapshot()

    def rgbd_frame(self) -> RGBDFrame:
        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError(
                    "the simulation thread failed"
                ) from self._thread_error
            if self._latest_frame is None:
                raise RuntimeError("the simulator has not rendered a frame yet")
            return self._latest_frame

    def captured_frame(self):
        return self.rgbd_frame().captured_frame()

    def video_frame(self) -> SimulationVideoFrame:
        """Return the latest synchronized task and overview RGB views."""

        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError("the simulation thread failed") from self._thread_error
            if self._latest_video_frame is None:
                raise RuntimeError("the simulator has not rendered a video frame yet")
            return self._latest_video_frame

    def evidence_frame_pair(self) -> tuple[RGBDFrame, SimulationVideoFrame]:
        """Return one atomically selected RGB-D/video evidence pair."""

        with self._lock:
            if self._thread_error is not None:
                raise RuntimeError("the simulation thread failed") from self._thread_error
            robot = self._latest_frame
            video = self._latest_video_frame
            if robot is None or video is None:
                raise RuntimeError("the simulator has not rendered an evidence pair yet")
            if robot.sequence != video.sequence:
                raise RuntimeError("the simulator retained an unsynchronized evidence pair")
            return robot, video

    def wait_for_frame(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> RGBDFrame:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while (
                self._latest_frame is None
                or self._latest_frame.sequence <= after_sequence
            ):
                if self._thread_error is not None:
                    raise RuntimeError("the simulation thread failed") from self._thread_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a simulation frame")
                self._frame_condition.wait(remaining)
            return self._latest_frame

    def wait_for_video_frame(
        self,
        *,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> SimulationVideoFrame:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while (
                self._latest_video_frame is None
                or self._latest_video_frame.sequence <= after_sequence
            ):
                if self._thread_error is not None:
                    raise RuntimeError("the simulation thread failed") from self._thread_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a simulation video frame")
                self._frame_condition.wait(remaining)
            return self._latest_video_frame

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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[list(self._arm_qpos_addresses)] = HOME_QPOS[:7]
        self.data.qpos[list(self._finger_qpos_addresses)] = HOME_QPOS[7:]
        self._control_target[:7] = HOME_QPOS[:7]
        self._control_target[7] = 0.04
        self.data.ctrl[:] = self._control_target
        positions = self._sample_task_positions()
        for name, point in zip(TASK_BLOCKS, positions, strict=True):
            address = self._freejoint_qpos_address(name)
            self.data.qpos[address : address + 3] = (*point, 0.026)
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        for name, point in zip(
            STORED_BLOCKS,
            ((-1.30, -0.72), (-1.15, -0.72), (-1.00, -0.72)),
            strict=True,
        ):
            address = self._freejoint_qpos_address(name)
            self.data.qpos[address : address + 3] = (*point, 0.056)
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._latest_frame = None
        self._latest_video_frame = None

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
        renderer = None
        viewer = None
        try:
            renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
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
            wall_start = time.monotonic() - float(self.data.time)
            while not self._stop_event.is_set():
                with self._lock:
                    self.data.ctrl[:] = self._control_target
                    mujoco.mj_step(self.model, self.data)
                    if self.data.time >= next_render:
                        self._render_locked(renderer)
                        next_render = self.data.time + render_period
                    if viewer is not None:
                        if viewer.is_running():
                            viewer.sync()
                        else:
                            viewer.close()
                            viewer = None
                    simulation_time = float(self.data.time)
                if self.realtime:
                    now = time.monotonic()
                    delay = wall_start + simulation_time - now
                    if delay < -render_period:
                        # A renderer, driver, or host scheduling stall must not
                        # be followed by a burst of historical simulation
                        # frames labelled as contemporary camera acquisitions.
                        # Rebase the realtime epoch and continue from the
                        # current physical state at ordinary rate.  The missing
                        # wall interval remains visible as an evidence-source
                        # gap and therefore still fails the recorder threshold
                        # when it is materially long.
                        wall_start = now - simulation_time
                        delay = 0.0
                    if delay > 0:
                        self._stop_event.wait(min(delay, 0.01))
        except BaseException as error:
            with self._frame_condition:
                self._thread_error = error
                self._frame_condition.notify_all()
        finally:
            if viewer is not None:
                viewer.close()
            if renderer is not None:
                renderer.close()

    def _configure_viewer(self, viewer) -> None:
        """Configure only the interactive view; task rendering is independent."""

        if self.viewer_camera == "task":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self._camera_id
            viewer.opt.geomgroup[2] = 0
            return
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = (0.0, -0.15, 0.15)
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -28
        viewer.opt.geomgroup[2] = 1

    def _render_locked(self, renderer: mujoco.Renderer) -> None:
        renderer.disable_depth_rendering()

        renderer.update_scene(
            self.data,
            camera=self._overview_camera,
            scene_option=self._overview_scene_option,
        )
        overview = renderer.render().copy()
        renderer.update_scene(
            self.data,
            camera=self._camera_id,
            scene_option=self._task_scene_option,
        )
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(
            self.data,
            camera=self._camera_id,
            scene_option=self._task_scene_option,
        )
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()

        rotation = self.data.cam_xmat[self._camera_id].reshape(3, 3).copy()
        position = self.data.cam_xpos[self._camera_id].copy()
        calibration = CameraCalibration.from_fovy(
            width=self.width,
            height=self.height,
            fovy_degrees=float(self.model.cam_fovy[self._camera_id]),
            camera_position=position,
            camera_rotation=rotation,
        )
        self._frame_sequence += 1
        observed_at = time.monotonic()
        self._latest_frame = RGBDFrame(
            rgb=rgb,
            depth_m=depth,
            calibration=calibration,
            observed_at=observed_at,
            sequence=self._frame_sequence,
            simulation_time=float(self.data.time),
        )
        self._latest_video_frame = SimulationVideoFrame(
            sequence=self._frame_sequence,
            observed_at=observed_at,
            simulation_time=float(self.data.time),
            task_rgb=rgb,
            overview_rgb=overview,
        )
        self._frame_condition.notify_all()


__all__ = [
    "ALL_BLOCKS",
    "HOME_QPOS",
    "InMemoryTaskPublisher",
    "SimulationSnapshot",
    "SimulationVideoFrame",
    "StackingEnvironment",
]
