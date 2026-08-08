"""Synchronized RGB-D observation contracts and camera geometry."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from prefmem.agents.monitor import CapturedFrame


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    width: int
    height: int
    intrinsic: np.ndarray
    world_from_camera: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        intrinsic = _array(self.intrinsic, (3, 3), "intrinsic")
        transform = _array(self.world_from_camera, (4, 4), "world_from_camera")
        if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("world_from_camera must be a homogeneous transform")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "world_from_camera", transform)

    @classmethod
    def from_fovy(
        cls,
        *,
        width: int,
        height: int,
        fovy_degrees: float,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
    ) -> CameraCalibration:
        if not math.isfinite(fovy_degrees) or not 0 < fovy_degrees < 180:
            raise ValueError("fovy_degrees must be between 0 and 180")
        fy = 0.5 * height / math.tan(math.radians(fovy_degrees) / 2.0)
        fx = fy
        intrinsic = np.array(
            [
                [fx, 0.0, (width - 1) / 2.0],
                [0.0, fy, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(camera_rotation).reshape(3, 3)
        transform[:3, 3] = np.asarray(camera_position).reshape(3)
        return cls(width, height, intrinsic, transform)

    def backproject(self, pixels_uv: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Back-project metric optical-axis depth into MuJoCo world coordinates."""

        pixels = np.asarray(pixels_uv, dtype=np.float64)
        depths = np.asarray(depth_m, dtype=np.float64).reshape(-1)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixels_uv must have shape (N, 2)")
        if pixels.shape[0] != depths.shape[0]:
            raise ValueError("pixels_uv and depth_m must contain the same number")
        if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(depths)):
            raise ValueError("pixels and depth values must be finite")
        if np.any(depths <= 0):
            raise ValueError("depth values must be positive")
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        x = (pixels[:, 0] - cx) * depths / fx
        # Image rows grow downward, while the MuJoCo camera's local +Y is up.
        y = -(pixels[:, 1] - cy) * depths / fy
        # MuJoCo cameras look along their local -Z axis.
        camera_points = np.column_stack((x, y, -depths))
        rotation = self.world_from_camera[:3, :3]
        translation = self.world_from_camera[:3, 3]
        return camera_points @ rotation.T + translation


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    calibration: CameraCalibration
    observed_at: float
    sequence: int
    simulation_time: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CameraCalibration):
            raise ValueError("calibration must be CameraCalibration")
        rgb = np.asarray(self.rgb)
        expected_rgb = (self.calibration.height, self.calibration.width, 3)
        if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 with shape {expected_rgb}")
        depth = np.asarray(self.depth_m, dtype=np.float32)
        expected_depth = (self.calibration.height, self.calibration.width)
        if depth.shape != expected_depth:
            raise ValueError(f"depth_m must have shape {expected_depth}")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be a non-negative integer")
        if self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if isinstance(self.observed_at, bool) or not isinstance(
            self.observed_at, (int, float)
        ) or not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")
        if self.simulation_time is not None and (
            isinstance(self.simulation_time, bool)
            or not isinstance(self.simulation_time, (int, float))
            or not math.isfinite(self.simulation_time)
        ):
            raise ValueError("simulation_time must be finite or None")
        rgb = rgb.copy()
        depth = depth.copy()
        rgb.setflags(write=False)
        depth.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "observed_at", float(self.observed_at))
        if self.simulation_time is not None:
            object.__setattr__(self, "simulation_time", float(self.simulation_time))

    def jpeg_bytes(self, *, quality: int = 95) -> bytes:
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError("could not encode the simulation frame as JPEG")
        return encoded.tobytes()

    def image_block(self) -> dict[str, Any]:
        encoded = base64.b64encode(self.jpeg_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        }

    def captured_frame(self) -> CapturedFrame:
        return CapturedFrame(
            image_block=self.image_block(),
            observed_at=self.observed_at,
            sequence=self.sequence,
        )
