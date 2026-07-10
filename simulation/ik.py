"""Differential inverse kinematics for the ARX L5 arm (damped least squares).

    ik = DiffIK(model)
    q_cmd = data.qpos[ik.qadr].copy()
    while running:
        q_cmd = ik.step(data, target_pos, target_quat, q_cmd)
        data.ctrl[:6] = q_cmd
        mujoco.mj_step(model, data)

`step` integrates the *command*, not the measured position -- see its docstring.
"""

import mujoco
import numpy as np

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

EE_BODY = "link6"

# Grasp center between the two finger pads, expressed in the link6 frame.
TCP_OFFSET = np.array([0.1366, 0.0, 0.0])


class DiffIK:
    """Maps a tool-frame pose error to arm joint velocities.

    Resolves the rate through a damped pseudo-inverse,

        dq = J^T (J J^T + damping^2 I)^-1 @ twist,

    trading tracking accuracy for conditioning near singularities. Solves over
    the six arm joints only, so the fingers keep whatever the gripper actuator
    commands. Errors decay with time constant ``1/gain``.
    """

    def __init__(self, model, *, damping=1e-2, max_speed=np.pi, pos_gain=1.0, ori_gain=1.0):
        self.model = model
        self.damping = damping
        self.max_speed = max_speed
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain

        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS])
        if (ids == -1).any():
            raise ValueError(f"model is missing arm joints {ARM_JOINTS}")
        self.qadr = model.jnt_qposadr[ids]
        self.dofadr = model.jnt_dofadr[ids]
        self.limited = model.jnt_limited[ids].astype(bool)
        self.lo = model.jnt_range[ids[self.limited], 0]
        self.hi = model.jnt_range[ids[self.limited], 1]

        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def tool_pose(self, data):
        """World (pos, quat) of the tool center point. Quat is (w, x, y, z)."""
        xmat = data.xmat[self.body].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, xmat.ravel())
        return data.xpos[self.body] + xmat @ TCP_OFFSET, quat

    def error(self, data, target_pos, target_quat=None):
        """6D twist (v, omega) in world frame carrying the tool onto the target.

        Pass ``target_quat=None`` to leave orientation free; the angular part is
        then left at zero.
        """
        tool_pos, tool_quat = self.tool_pose(data)

        err = np.zeros(6)
        err[:3] = np.asarray(target_pos, dtype=float) - tool_pos
        if target_quat is None:
            return err

        tgt = np.array(target_quat, dtype=float)
        mujoco.mju_normalize4(tgt)

        # world-frame residual rotation: target * conj(tool), matching mj_jac's frame
        conj, residual = np.empty(4), np.empty(4)
        mujoco.mju_negQuat(conj, tool_quat)
        mujoco.mju_mulQuat(residual, tgt, conj)
        mujoco.mju_quat2Vel(err[3:], residual, 1.0)
        return err

    def velocity(self, data, target_pos, target_quat=None, feedforward=None,
                 max_linear_speed=None, max_angular_speed=None):
        """Arm joint velocities (rad/s) driving the tool toward the target pose.

        ``feedforward`` is an optional desired Cartesian twist ``(v, omega)``.
        The pose-error term then acts as tracking feedback instead of being solely
        responsible for generating motion.  Omitting it preserves the original
        fixed-target servo behaviour.

        Scaled down as a group if any joint exceeds ``max_speed``, so the
        commanded direction is preserved.
        """
        err = self.error(data, target_pos, target_quat)
        twist = np.concatenate([self.pos_gain * err[:3], self.ori_gain * err[3:]])
        if feedforward is not None:
            ff = np.asarray(feedforward, dtype=float)
            if ff.shape != (6,):
                raise ValueError(f"feedforward twist must have shape (6,), got {ff.shape}")
            twist += ff
        for part, limit in ((twist[:3], max_linear_speed),
                            (twist[3:], max_angular_speed)):
            norm = np.linalg.norm(part)
            if limit is not None and norm > limit:
                part *= float(limit) / norm

        mujoco.mj_jac(self.model, data, self._jacp, self._jacr, self.tool_pose(data)[0], self.body)
        jac = np.vstack([self._jacp, self._jacr])[:, self.dofadr]

        if target_quat is None:  # position-only task: drop the rotational rows
            jac, twist = jac[:3], twist[:3]

        reg = (self.damping**2) * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + reg, twist)

        scale = np.max(np.abs(dq)) / self.max_speed
        return dq / scale if scale > 1.0 else dq

    def integrate(self, q_cmd, dq, dt=None):
        """Integrate a joint-velocity command and clamp it to joint limits."""
        q = np.asarray(q_cmd, dtype=float) + np.asarray(dq, dtype=float) * (
            self.model.opt.timestep if dt is None else dt)
        q[self.limited] = np.clip(q[self.limited], self.lo, self.hi)
        return q

    def step(self, data, target_pos, target_quat, q_cmd, dt=None, feedforward=None,
             max_linear_speed=None, max_angular_speed=None):
        """Integrate one velocity step onto ``q_cmd`` and clamp to joint limits.

        Returns the next position setpoint for the arm actuators, which are
        position servos. Feed the return value back in as ``q_cmd``: integrating
        the *command* is what makes the loop converge. Rebuilding the setpoint
        from the measured ``data.qpos`` instead folds the servo's steady-state
        gravity sag (~0.07 rad; links 1-6 have no gravcomp) back into the command
        every step, so the loop settles where ``dq*dt`` cancels the sag rather
        than where the pose error is zero -- centimeters of residual. Integrating
        the command makes ``dq -> 0`` the only fixed point, and the servo absorbs
        the sag.

        ``dt`` defaults to the physics timestep, right for one call per mj_step.
        """
        dq = self.velocity(
            data, target_pos, target_quat, feedforward=feedforward,
            max_linear_speed=max_linear_speed, max_angular_speed=max_angular_speed)
        return self.integrate(q_cmd, dq, dt)
