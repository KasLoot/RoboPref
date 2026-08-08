"""Deterministic Cartesian waypoint controller for the MuJoCo Panda."""

from __future__ import annotations

import math
import threading
import time

import mujoco
import numpy as np

from prefmem.execution.contracts import (
    AnchorKind,
    GripperCommand,
    ManipulationProgram,
    WaypointKind,
)
from simulation.stacking import HOME_QPOS, StackingEnvironment


# The quintic minimum-jerk curve has these normalized extrema.  Including
# them in the duration calculation makes the public speed/acceleration values
# real limits instead of average-rate hints.
_MINIMUM_JERK_PEAK_VELOCITY = 1.875
_MINIMUM_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)
_MINIMUM_MOTION_SECONDS = 0.65


class InverseKinematicsError(RuntimeError):
    """A requested end-effector destination is not safely reachable."""


class MotionControlError(RuntimeError):
    """The simulated robot could not settle at a commanded target."""


class PandaPickPlaceController:
    """IK followed by gravity-compensated minimum-jerk joint control."""

    def __init__(
        self,
        environment: StackingEnvironment,
        *,
        control_hz: float = 50.0,
        joint_speed_radians: float = 0.6,
        joint_acceleration_radians: float = 1.2,
        grasp_offset_m: float = 0.024,
        placement_offset_m: float = 0.027,
    ) -> None:
        if not isinstance(environment, StackingEnvironment):
            raise TypeError("environment must be a StackingEnvironment")
        tuning_values = (
            control_hz,
            joint_speed_radians,
            joint_acceleration_radians,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in tuning_values
        ):
            raise ValueError("control rates must be positive")
        self.environment = environment
        self.model = environment.model
        self.control_period = 1.0 / float(control_hz)
        self.joint_speed_radians = float(joint_speed_radians)
        self.joint_acceleration_radians = float(
            joint_acceleration_radians
        )
        self.grasp_offset_m = float(grasp_offset_m)
        self.placement_offset_m = float(placement_offset_m)
        self.settle_position_tolerance_radians = 0.008
        self.settle_velocity_tolerance_radians = 0.05
        self.settle_stability_seconds = 0.20
        self.settle_timeout_seconds = 3.0
        self._command_lock = threading.RLock()
        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )
        if self._site_id < 0:
            raise RuntimeError("Panda model has no gripper site")
        self._joint_ids = tuple(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}"
            )
            for index in range(1, 8)
        )
        if any(item < 0 for item in self._joint_ids):
            raise RuntimeError("Panda model is missing an arm joint")
        self._qpos_addresses = np.array(
            [self.model.jnt_qposadr[item] for item in self._joint_ids], dtype=int
        )
        self._dof_addresses = np.array(
            [self.model.jnt_dofadr[item] for item in self._joint_ids], dtype=int
        )
        self._joint_ranges = self.model.jnt_range[list(self._joint_ids)].copy()
        self._actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"actuator{index}",
                )
                for index in range(1, 8)
            ],
            dtype=int,
        )
        if np.any(self._actuator_ids < 0):
            raise RuntimeError("Panda model is missing an arm actuator")
        if not np.array_equal(self._actuator_ids, np.arange(7)):
            raise RuntimeError(
                "Panda arm actuators must be the first seven controls"
            )
        self._actuator_kp = self.model.actuator_gainprm[
            self._actuator_ids, 0
        ].copy()
        if np.any(self._actuator_kp <= 0):
            raise RuntimeError("Panda arm actuators need positive position gains")
        self._actuator_force_ranges = self.model.actuator_forcerange[
            self._actuator_ids
        ].copy()
        self._actuator_force_limited = self.model.actuator_forcelimited[
            self._actuator_ids
        ].astype(bool)
        self._actuator_control_ranges = self.model.actuator_ctrlrange[
            self._actuator_ids
        ].copy()
        self._gravity_data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self._gravity_data)
        self._gravity_data.qpos[:] = self.model.qpos0
        home_data = mujoco.MjData(self.model)
        home_data.qpos[self._qpos_addresses] = HOME_QPOS[:7]
        mujoco.mj_forward(self.model, home_data)
        self._tool_down_rotation = home_data.site_xmat[self._site_id].reshape(3, 3).copy()

    def solve_ik(
        self,
        position_world: np.ndarray,
        *,
        seed: np.ndarray | None = None,
        maximum_iterations: int = 250,
    ) -> np.ndarray:
        target = np.asarray(position_world, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("position_world must contain three finite values")
        if not (
            0.15 <= target[0] <= 0.85
            and abs(target[1]) <= 0.45
            and 0.01 <= target[2] <= 0.65
        ):
            raise InverseKinematicsError(
                f"destination lies outside the configured Panda workspace: {target}"
            )
        q = (
            self.environment.arm_qpos()
            if seed is None
            else np.asarray(seed, dtype=np.float64).copy()
        )
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("IK seed must contain seven finite values")
        q = np.clip(q, self._joint_ranges[:, 0], self._joint_ranges[:, 1])
        scratch = mujoco.MjData(self.model)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)

        last_position_error = math.inf
        last_rotation_error = math.inf
        for _ in range(maximum_iterations):
            scratch.qpos[self._qpos_addresses] = q
            mujoco.mj_forward(self.model, scratch)
            current_position = scratch.site_xpos[self._site_id]
            current_rotation = scratch.site_xmat[self._site_id].reshape(3, 3)
            position_error = target - current_position
            rotation_error = 0.5 * sum(
                (
                    np.cross(current_rotation[:, index], self._tool_down_rotation[:, index])
                    for index in range(3)
                ),
                start=np.zeros(3, dtype=np.float64),
            )
            last_position_error = float(np.linalg.norm(position_error))
            last_rotation_error = float(np.linalg.norm(rotation_error))
            if last_position_error < 0.0015 and last_rotation_error < 0.025:
                return q.copy()
            jacobian_position.fill(0.0)
            jacobian_rotation.fill(0.0)
            mujoco.mj_jacSite(
                self.model,
                scratch,
                jacobian_position,
                jacobian_rotation,
                self._site_id,
            )
            position_jacobian = jacobian_position[:, self._dof_addresses]
            rotation_jacobian = jacobian_rotation[:, self._dof_addresses]
            orientation_weight = 0.30
            jacobian = np.vstack(
                (position_jacobian, orientation_weight * rotation_jacobian)
            )
            error = np.concatenate(
                (position_error, orientation_weight * rotation_error)
            )
            damping = 0.035
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            update_norm = float(np.linalg.norm(update))
            if update_norm > 0.12:
                update *= 0.12 / update_norm
            # A weak home regularizer avoids joint-limit solutions without
            # materially moving the Cartesian destination.
            update += 0.001 * (HOME_QPOS[:7] - q)
            q = np.clip(q + update, self._joint_ranges[:, 0], self._joint_ranges[:, 1])

        raise InverseKinematicsError(
            "IK did not converge for destination "
            f"{target.tolist()} (position error {last_position_error:.4f} m, "
            f"rotation error {last_rotation_error:.4f} rad)"
        )

    def execute_pick_place(
        self,
        program: ManipulationProgram,
        source_world: np.ndarray,
        target_world: np.ndarray,
        cancel_event: threading.Event,
    ) -> None:
        if not isinstance(program, ManipulationProgram):
            raise TypeError("program must be a ManipulationProgram")
        if not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event")
        source = np.asarray(source_world, dtype=np.float64)
        target = np.asarray(target_world, dtype=np.float64)
        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("source and target must contain three coordinates")

        grasp = source.copy()
        grasp[2] -= self.grasp_offset_m
        place = target.copy()
        place[2] += self.placement_offset_m
        waypoint_positions: dict[WaypointKind, np.ndarray] = {}
        for waypoint in program.waypoints:
            if waypoint.kind is WaypointKind.APPROACH_SOURCE:
                point = grasp + np.array([0.0, 0.0, waypoint.clearance_m])
            elif waypoint.kind is WaypointKind.GRASP_SOURCE:
                point = grasp
            elif waypoint.kind is WaypointKind.LIFT:
                point = grasp + np.array([0.0, 0.0, waypoint.clearance_m])
            elif waypoint.kind is WaypointKind.APPROACH_TARGET:
                point = place + np.array([0.0, 0.0, waypoint.clearance_m])
            elif waypoint.kind is WaypointKind.PLACE_TARGET:
                point = place
            else:
                point = place + np.array([0.0, 0.0, waypoint.clearance_m])
            waypoint_positions[waypoint.kind] = point

        with self._command_lock:
            self._check_cancel(cancel_event)
            seed = self.environment.arm_qpos()
            joint_targets: dict[WaypointKind, np.ndarray] = {}
            for waypoint in program.waypoints:
                self._check_cancel(cancel_event)
                seed = self.solve_ik(waypoint_positions[waypoint.kind], seed=seed)
                joint_targets[waypoint.kind] = seed

            for waypoint in program.waypoints:
                self._check_cancel(cancel_event)
                try:
                    self._move_arm(joint_targets[waypoint.kind], cancel_event)
                except MotionControlError as error:
                    raise MotionControlError(
                        f"{waypoint.kind.value}: {error}"
                    ) from error
                if waypoint.gripper is GripperCommand.OPEN:
                    self.environment.set_control(gripper=0.04)
                    self._wait(0.45, cancel_event)
                elif waypoint.gripper is GripperCommand.CLOSE:
                    self.environment.set_control(gripper=0.0)
                    self._wait(0.8, cancel_event)

            # Park clear of the fixed overhead task camera before allowing the
            # Monitor to make a terminal judgment.
            try:
                self._move_arm(HOME_QPOS[:7], cancel_event)
            except MotionControlError as error:
                raise MotionControlError(f"park: {error}") from error

    def safe_hold(self) -> None:
        with self._command_lock:
            snapshot = self.environment.snapshot()
            arm = self._gravity_compensated_control(
                snapshot.qpos[self._qpos_addresses]
            )
            # Preserve the active open/close command. Commanding the measured
            # finger gap would remove grasp force and could drop a held cube.
            gripper = float(np.clip(snapshot.control[7], 0.0, 0.04))
            self.environment.set_control(arm=arm, gripper=gripper)

    def _move_arm(
        self,
        target: np.ndarray,
        cancel_event: threading.Event,
    ) -> None:
        target = np.asarray(target, dtype=np.float64)
        start = self.environment.arm_qpos()
        start_control = self.environment.snapshot().control[:7].copy()
        maximum_delta = float(np.max(np.abs(target - start)))
        duration = self._motion_duration(maximum_delta)
        steps = max(2, int(math.ceil(duration / self.control_period)))
        start_compensation = start_control - start
        for index in range(1, steps + 1):
            self._check_cancel(cancel_event)
            fraction = index / steps
            smooth = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
            desired = start + smooth * (target - start)
            compensated = self._gravity_compensated_control(desired)
            # Blend the existing hold offset into the model-based gravity
            # offset so the first control sample cannot jump.
            compensation = (
                (1.0 - smooth) * start_compensation
                + smooth * (compensated - desired)
            )
            command = np.clip(
                desired + compensation,
                self._actuator_control_ranges[:, 0],
                self._actuator_control_ranges[:, 1],
            )
            self.environment.set_control(arm=command)
            if cancel_event.wait(self.control_period):
                self.safe_hold()
                raise InterruptedError("execution was cancelled during motion")

        # Hold with model-based gravity feed-forward.  The former accumulating
        # trim could wind the command past the target while the physical arm
        # was still catching up, which caused visible overshoot.
        command = self._gravity_compensated_control(target)
        self.environment.set_control(arm=command)
        deadline = time.monotonic() + self.settle_timeout_seconds
        stable_since: float | None = None
        while time.monotonic() < deadline:
            self._check_cancel(cancel_event)
            actual, velocity = self.environment.arm_state()
            joint_error = target - actual
            position_error = float(np.max(np.abs(joint_error)))
            speed = float(np.max(np.abs(velocity)))
            if (
                position_error <= self.settle_position_tolerance_radians
                and speed <= self.settle_velocity_tolerance_radians
            ):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif (
                    time.monotonic() - stable_since
                    >= self.settle_stability_seconds
                ):
                    return
            else:
                stable_since = None
            self.environment.set_control(arm=command)
            if cancel_event.wait(self.control_period):
                break
        self.safe_hold()
        if cancel_event.is_set():
            raise InterruptedError("execution was cancelled while settling")
        final_qpos, final_qvel = self.environment.arm_state()
        final_error = float(np.max(np.abs(final_qpos - target)))
        final_speed = float(np.max(np.abs(final_qvel)))
        raise MotionControlError(
            "Panda arm did not settle at its joint target "
            f"(maximum error {final_error:.3f} rad, maximum speed "
            f"{final_speed:.3f} rad/s)"
        )

    def _motion_duration(self, maximum_delta: float) -> float:
        """Duration required to respect minimum-jerk speed and acceleration."""

        if not math.isfinite(maximum_delta) or maximum_delta < 0:
            raise ValueError("maximum_delta must be finite and non-negative")
        speed_duration = (
            _MINIMUM_JERK_PEAK_VELOCITY
            * maximum_delta
            / self.joint_speed_radians
        )
        acceleration_duration = math.sqrt(
            _MINIMUM_JERK_PEAK_ACCELERATION
            * maximum_delta
            / self.joint_acceleration_radians
        )
        return max(
            _MINIMUM_MOTION_SECONDS,
            speed_duration,
            acceleration_duration,
        )

    def _gravity_compensated_control(
        self,
        desired_qpos: np.ndarray,
    ) -> np.ndarray:
        """Convert a desired arm pose to a non-winding position setpoint."""

        desired = np.asarray(desired_qpos, dtype=np.float64)
        if desired.shape != (7,) or not np.all(np.isfinite(desired)):
            raise ValueError("desired_qpos must contain seven finite values")
        self._gravity_data.qpos[self._qpos_addresses] = desired
        self._gravity_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._gravity_data)
        required_effort = self._gravity_data.qfrc_bias[
            self._dof_addresses
        ].copy()
        limited = self._actuator_force_limited
        required_effort[limited] = np.clip(
            required_effort[limited],
            self._actuator_force_ranges[limited, 0],
            self._actuator_force_ranges[limited, 1],
        )
        command = desired + required_effort / self._actuator_kp
        return np.clip(
            command,
            self._actuator_control_ranges[:, 0],
            self._actuator_control_ranges[:, 1],
        )

    @staticmethod
    def _wait(duration: float, cancel_event: threading.Event) -> None:
        if cancel_event.wait(duration):
            raise InterruptedError("execution was cancelled during gripper motion")

    @staticmethod
    def _check_cancel(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise InterruptedError("execution was cancelled")


__all__ = [
    "InverseKinematicsError",
    "MotionControlError",
    "PandaPickPlaceController",
]
