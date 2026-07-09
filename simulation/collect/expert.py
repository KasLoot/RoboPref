"""Scripted pick-and-place expert on top of the frozen DiffIK + CriticalDamper.

The inner loop is engine.servo_to's, unchanged in shape:

    damper.apply(data)
    q_cmd = ik.step(data, target_pos, target_quat, q_cmd)
    data.ctrl[:6] = q_cmd
    mujoco.mj_step(model, data)

The command is integrated and fed back, never rebuilt from measured qpos.

Recording hooks into ``on_step``. MuJoCo's ``mj_step`` runs its forward pass at the
pre-step qpos and only then integrates, so *after* the call ``data.xpos`` and
``data.qfrc_actuator`` describe the state at the step's start while ``data.time``
has already advanced. Caching qpos/qvel before the step and reading the rest after
it therefore yields one consistent snapshot at ``t = data.time - timestep``.
"""

import mujoco
import numpy as np

import config as C
from ik import DiffIK
from servo import CriticalDamper


class ExpertFailure(RuntimeError):
    """A waypoint timed out, or a target fell outside the verified workspace."""


def safe_transit_z(top_level):
    """TCP height for every horizontal move in an episode whose tallest stack ends
    at ``top_level`` (0 = a single block on the mat).

    A carried block hangs BLOCK_HALF + GRASP_DZ below the TCP, 4.5 mm below the
    finger tips, so this is the binding clearance. Moving diagonally off a freshly
    placed block instead of retreating to this height drags the open fingers across
    the stack top and topples it -- that is what breaks a 4-high build.
    """
    hover = C.block_center_z(top_level) + C.RELEASE_GAP + C.GRASP_DZ + C.HOVER_DZ
    stack_top = C.block_center_z(max(top_level - 1, 0)) + C.BLOCK_HALF
    clearance = stack_top + C.BLOCK_HALF + C.GRASP_DZ + C.TRANSIT_CLEARANCE
    pregrasp = C.block_center_z(0) + C.GRASP_DZ + C.PREGRASP_DZ
    return max(hover, clearance, pregrasp)


class Expert:
    def __init__(self, scene, data, on_step=None, safe_z=None):
        self.scene = scene
        self.data = data
        self.model = scene.model
        self.on_step = on_step
        self.safe_z = safe_transit_z(0) if safe_z is None else float(safe_z)

        self.ik = DiffIK(self.model, pos_gain=C.GAIN, ori_gain=C.GAIN)
        self.damper = CriticalDamper(self.model)
        self.q_cmd = data.qpos[self.ik.qadr].copy()
        self.grip = C.GRIP_OPEN
        self._carry = None  # (TCP - block) offset measured on the held block
        self.target_pos = None  # last servo target, for the preview's markers
        self.target_quat = None

        if self.scene.gripper_act != 6 or self.model.nu != 7:
            raise RuntimeError("expected actuators 0..5 = arm, 6 = gripper")

    # -- physics ------------------------------------------------------------
    def _step(self, target_pos=None, target_quat=None):
        model, data = self.model, self.data
        qpos = data.qpos[self.ik.qadr].copy()
        qvel = data.qvel[self.ik.dofadr].copy()

        self.damper.apply(data)
        if target_pos is not None:
            self.target_pos, self.target_quat = target_pos, target_quat
            self.q_cmd = self.ik.step(data, target_pos, target_quat, self.q_cmd)
        data.ctrl[:6] = self.q_cmd
        data.ctrl[self.scene.gripper_act] = self.grip
        mujoco.mj_step(model, data)

        if self.on_step is not None:
            ee_pos, ee_quat = self.ik.tool_pose(data)
            self.on_step(
                t=data.time - model.opt.timestep,
                qpos=qpos,
                qvel=qvel,
                qtau=data.qfrc_actuator[self.ik.dofadr].copy(),
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                q_cmd=self.q_cmd,
                grip=self.grip,
                data=data,
            )

    def hold(self, seconds):
        """Freeze the command and let physics run (settling, gripper motion)."""
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            self._step()

    def set_grip(self, value, seconds):
        self.grip = float(value)
        self.hold(seconds)

    def servo(self, target_pos, target_quat, tight):
        """Drive the tool onto a pose. Raises ExpertFailure on timeout."""
        target_pos = np.asarray(target_pos, dtype=float)
        if not C.reachable(target_pos):
            raise ExpertFailure(f"target {target_pos} outside the verified workspace")

        pos_tol, ori_tol, hold = ((C.POS_TOL, C.ORI_TOL, C.HOLD) if tight else
                                  (C.POS_TOL_TRANSIT, C.ORI_TOL_TRANSIT, C.HOLD_TRANSIT))
        settled, deadline = 0.0, self.data.time + C.TIMEOUT
        while self.data.time < deadline:
            self._step(target_pos, target_quat)
            err = self.ik.error(self.data, target_pos, target_quat)
            pos_err, ori_err = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            inside = pos_err < pos_tol and ori_err < ori_tol
            settled = settled + self.model.opt.timestep if inside else 0.0
            if settled >= hold:
                return
        raise ExpertFailure(
            f"timeout at {target_pos} (pos_err={pos_err * 1e3:.2f} mm, "
            f"ori_err={np.rad2deg(ori_err):.2f} deg)")

    # -- primitives ---------------------------------------------------------
    def pick(self, color):
        """Pregrasp above the block, descend, close, lift. Returns the grasp quat.

        The block pose is re-read from ``data`` at call time: earlier placements and
        settling move things. After the lift we measure where the block actually
        ended up in the fingers -- closing them nudges it by ~0.7 mm, and it can
        settle a little low -- and ``place`` corrects the target by that offset.
        """
        pos = self.scene.block_pos(self.data, color)
        quat = C.tool_down_quat(C.grasp_yaw(self.scene.block_yaw(self.data, color)))

        grasp = np.array([pos[0], pos[1], pos[2] + C.GRASP_DZ])
        pregrasp = grasp + [0.0, 0.0, C.PREGRASP_DZ]
        safe = np.array([pos[0], pos[1], max(self.safe_z, grasp[2] + C.LIFT_DZ)])

        self.set_grip(C.GRIP_OPEN, 0.0)
        self.servo(safe, quat, tight=False)  # cross the scene high, then come down
        self.servo(pregrasp, quat, tight=False)
        self.servo(grasp, quat, tight=True)
        self.set_grip(C.GRIP_CLOSE, C.GRIP_CLOSE_TIME)
        self.servo(safe, quat, tight=False)  # straight up, never diagonally

        tcp = self.ik.tool_pose(self.data)[0]
        self._carry = tcp - self.scene.block_pos(self.data, color)
        return quat

    def place(self, target_xy, target_z, quat):
        """Transit above the target, descend, open, retreat.

        ``target_xy``/``target_z`` are where the *block centre* should come to rest.
        The TCP target is offset by the measured carry offset, so the block -- not
        the tool -- lands on the target. Without this the tower leans: a few mm of
        bias per level compounds and the stack topples at four high.
        """
        if self._carry is None:
            raise ExpertFailure("place() called without a preceding pick()")

        place_pos = np.array([target_xy[0] + self._carry[0],
                              target_xy[1] + self._carry[1],
                              target_z + C.RELEASE_GAP + self._carry[2]])
        hover = place_pos + [0.0, 0.0, C.HOVER_DZ]
        safe = np.array([place_pos[0], place_pos[1], max(self.safe_z, hover[2])])

        self.servo(safe, quat, tight=False)  # arrive over the target from above
        self.servo(hover, quat, tight=False)
        self.servo(place_pos, quat, tight=True)
        self.set_grip(C.GRIP_OPEN, C.GRIP_OPEN_TIME)
        self.servo(safe, quat, tight=False)  # straight up, clear of the stack
        self._carry = None

    def transfer(self, color, target_xy, target_z):
        quat = self.pick(color)
        self.place(target_xy, target_z, quat)


def stack_target(scene, data, below):
    """Where the next block goes: the measured centre of the stack top, one block up.

    ``below`` is the bottom->top list of colors already stacked. Empty means the
    cross itself. Always re-read from ``data``: the block below moved when it landed.
    """
    if not below:
        return scene.cross_xy.copy(), C.block_center_z(0)
    top = scene.block_pos(data, below[-1])
    return top[:2].copy(), float(top[2] + C.BLOCK_SIZE)


# ---------------------------------------------------------------------------
# success predicates -- ground truth, evaluated after SETTLE_TIME
# ---------------------------------------------------------------------------
def blocks_at_rest(scene, data):
    return all(scene.block_speed(data, p.color) < C.SPEED_TOL for p in scene.placements)


def on_cross(scene, data, color):
    """Base of the stack: within XY_TOL of the cross, resting at level 0."""
    pos = scene.block_pos(data, color)
    return (np.linalg.norm(pos[:2] - scene.cross_xy) <= C.XY_TOL
            and abs(pos[2] - C.block_center_z(0)) <= C.Z_TOL)


def stacked_on(scene, data, upper, lower, level):
    """``upper`` sits on ``lower``: xy aligned, and at the level's expected height."""
    up = scene.block_pos(data, upper)
    lo = scene.block_pos(data, lower)
    return (np.linalg.norm(up[:2] - lo[:2]) <= C.XY_TOL
            and abs(up[2] - C.block_center_z(level)) <= C.Z_TOL)


def verify_stack(scene, data, order):
    """Is the stack at the cross exactly ``order`` (bottom -> top) and at rest?

    Also confirms the *executed* order matches the intended one, by reading the
    resting heights rather than trusting the script.
    """
    if not blocks_at_rest(scene, data):
        return False, "blocks still moving"
    if not on_cross(scene, data, order[0]):
        return False, f"base {order[0]} not on the cross"
    for level in range(1, len(order)):
        if not stacked_on(scene, data, order[level], order[level - 1], level):
            return False, f"{order[level]} not stacked on {order[level - 1]}"

    heights = sorted(((scene.block_pos(data, c)[2], c) for c in order))
    executed = [c for _, c in heights]
    if executed != list(order):
        return False, f"executed order {executed} != intended {list(order)}"
    return True, "ok"


def verify_on_cross(scene, data, color):
    """Category C, T-MAT: one block delivered to the cross."""
    if not blocks_at_rest(scene, data):
        return False, "blocks still moving"
    if not on_cross(scene, data, color):
        return False, f"{color} not on the cross"
    return True, "ok"


def verify_on_top(scene, data, color, init_stack):
    """Category C, T-TOP: one block delivered onto an existing stack."""
    if not blocks_at_rest(scene, data):
        return False, "blocks still moving"
    level = len(init_stack)
    if level == 0:
        return verify_on_cross(scene, data, color)
    if not stacked_on(scene, data, color, init_stack[-1], level):
        return False, f"{color} not stacked on {init_stack[-1]}"
    # the stack underneath must not have been disturbed
    for lvl, c in enumerate(init_stack):
        pos = scene.block_pos(data, c)
        if np.linalg.norm(pos[:2] - scene.cross_xy) > C.XY_TOL:
            return False, f"stack block {c} shifted off the cross"
        if abs(pos[2] - C.block_center_z(lvl)) > C.Z_TOL:
            return False, f"stack block {c} moved in z"
    return True, "ok"
