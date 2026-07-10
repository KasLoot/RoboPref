"""Shared constants for the RoboPref collection suite.

Everything here is either measured off the compiled model (see README, "Phase 0")
or derived from something measured. This module is where scene geometry,
trajectory limits, tolerances and budgets live.

Importing this module puts ``simulation/`` on ``sys.path`` so that ``ik`` and
``servo`` -- which import each other by bare name -- resolve.
"""

import os
import sys
import platform
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = SIM_DIR.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

# Rendering backend. Must be set before MuJoCo creates a GL context.
# If MUJOCO_GL is set to an invalid value (for example, "eg"), normalise it
# to a platform-appropriate default so imports fail less often across OSes.
_SYSTEM = platform.system()
_MUJOCO_GL = os.environ.get("MUJOCO_GL", "").lower().strip()

_DISABLED = {"disable", "disabled", "off", "false", "0"}
_VALID = {"enable", "enabled", "on", "true", "1", "glfw", ""}
if _SYSTEM == "Linux":
    _VALID.update({"glx", "egl", "osmesa"})
elif _SYSTEM == "Windows":
    _VALID.update({"wgl"})
elif _SYSTEM == "Darwin":
    _VALID.update({"cgl"})

if _MUJOCO_GL not in _DISABLED and _MUJOCO_GL not in _VALID:
    if _SYSTEM == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
    elif _SYSTEM == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
    elif _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "glfw"
else:
    if _SYSTEM == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif _SYSTEM == "Windows":
        os.environ.setdefault("MUJOCO_GL", "wgl")
    elif _SYSTEM == "Darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")

# Linux-only: pick a DRI node for EGL.
if _SYSTEM == "Linux" and os.environ.get("MUJOCO_GL", "").lower().strip() == "egl":
    os.environ.setdefault("EGL_DEVICE_ID", "0")

SCENE_XML = str(SIM_DIR / "assets" / "robots" / "arx_l5" / "scene.xml")
DATASET_ROOT = Path("/data/datasets/RoboPref")

SCHEMA_VERSION = "1.0"
QUAT_CONVENTION = "wxyz"

# --------------------------------------------------------------------------
# Model facts (asserted against the compiled model at runtime, never assumed)
# --------------------------------------------------------------------------
EXPECTED_TIMESTEP = 0.002  # s -> 500 Hz proprio

GRIPPER_ACTUATOR = "gripper"  # the only actuator not driving joint1..joint6
GRIPPER_JOINT = "joint7"  # slide; joint8 is mirrored by an equality constraint
GRIPPER_CTRL_RANGE = (0.0, 0.044)

# Measured: inner-face separation of the finger pads is affine in the ctrl value.
#   gap(ctrl) = GRIP_GAP_AT_ZERO + 2 * ctrl      (87.8 mm wide open, 0.2 mm shut)
GRIP_GAP_AT_ZERO = 0.0002

# Measured: the lowest point of the gripper sits exactly this far below the TCP,
# independent of how far the fingers are open. This is what sets the grasp height.
GRIPPER_REACH_BELOW_TCP = 0.020

# Measured tool-down reach (radially symmetric about the base): the kinematic
# limit is r ~ 0.51 m, but below z ~ 0.026 the finger pads foul the floor and the
# reachable radius collapses (0.35 m at z = 0.0215). Everything the TCP visits
# must respect r_max(z).
R_MAX_AT_GRASP_Z = 0.51
R_MAX_BY_Z = ((0.0215, 0.350), (0.0245, 0.429), (0.0285, 0.510), (0.10, 0.477),
              (0.14, 0.445), (0.18, 0.400), (0.20, 0.364), (0.22, 0.299))

# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------
BLOCK_SIZE = 0.035  # m, cube edge
BLOCK_HALF = BLOCK_SIZE / 2
BLOCK_MASS = 0.030  # kg -- must be liftable by the ~0.7 N per-pad squeeze
BLOCK_FRICTION = (1.4, 0.02, 0.002)  # slide, torsion, roll
BLOCK_CONDIM = 4  # torsional friction: stacked blocks stop spinning off

COLORS = ("red", "green", "blue", "yellow")
COLOR_RGBA = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.70, 0.20, 1.0),
    "blue": (0.10, 0.25, 0.85, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
}

# --------------------------------------------------------------------------
# Mats and cross
#
# "left"/"right" are as seen by the third_person camera, which is the view a
# policy consumes: that camera's right axis is world +y, so world -y is image
# left. White mat -> image left (-y), black mat -> image right (+y).
# --------------------------------------------------------------------------
MAT_THICK = 0.004
MAT_TOP = MAT_THICK
# Sized so that four blocks at SPAWN_MIN_GAP fit on the black mat without the
# rejection sampler stalling, and so the cross region plus its arms sit well
# inside the white mat. Both mats stay inside r_max at every height the TCP visits.
MAT_HALF = (0.105, 0.095)  # x, y half-extents
WHITE_MAT_CENTER = (0.25, -0.135)
BLACK_MAT_CENTER = (0.25, +0.135)
WHITE_MAT_RGBA = (0.92, 0.92, 0.92, 1.0)
BLACK_MAT_RGBA = (0.07, 0.07, 0.07, 1.0)

CROSS_RGBA = (0.90, 0.10, 0.75, 1.0)  # magenta: distinct from all four block colors

# --------------------------------------------------------------------------
# Appearance overrides, applied to the spec per episode so the vendored robot
# asset (which ships its own LICENSE/CHANGELOG) is never edited.
# --------------------------------------------------------------------------
# Every visible robot geom shares the "black_mat" material. At its stock 0.1 grey the
# arm is nearly the same value as the black mat (0.07) and hard to segment against it.
ROBOT_MATERIAL = "black_mat"
ROBOT_RGBA = (0.55, 0.55, 0.55, 1.0)
# (The two red strips on link3 carry no material and stay red -- they are the arm's
# own markings, not part of the body colour.)

# The stock third_person camera sits at x = 1.05 aimed so its centre ray meets the
# ground at x = -0.29, *behind* the base: the mats sit at the bottom edge of the frame.
# Pushing it forward alone therefore crops them out. Move it forward AND aim it at the
# workspace. The camera stays on the y = 0 plane, so its right axis is still exactly
# world +y and the "white mat = image left" convention is unchanged.
#
#   THIRD_PERSON_POS = (distance from the base, 0, HEIGHT)
#                       ^ smaller = closer      ^ larger = higher
#
# Raising it tilts the view further down, which frames the mats better but walks the
# arm toward the top edge; past about z = 0.75 with x <= 0.68 the arm starts to clip.
THIRD_PERSON_POS = (0.75, 0.0, 0.65)
THIRD_PERSON_AIM = (0.0, 0.0, 0.06)
CROSS_ARM_HALF = 0.025  # 50 mm arms
CROSS_ARM_WIDTH = 0.003
CROSS_THICK = 0.0005  # flush on the mat; visual only (contype=conaffinity=0)
CROSS_REGION = 0.10  # side of the centred square the cross centre is drawn from

# Spawn margins on the black mat. The margin clears a block's half-diagonal
# (0.0175*sqrt2 = 24.7 mm) so a yawed block never overhangs the mat.
SPAWN_MARGIN = 0.028
SPAWN_HALF = (MAT_HALF[0] - SPAWN_MARGIN, MAT_HALF[1] - SPAWN_MARGIN)
# Spec floor is 1.5 * BLOCK_SIZE. We use 2.0: at 1.5 the open fingers (see
# GRIP_OPEN below) clear a neighbour by only ~5 mm on descent.
SPAWN_MIN_GAP = 2.0 * BLOCK_SIZE
SPAWN_MAX_TRIES = 500  # per block
SPAWN_MAX_RESTARTS = 40  # of the whole configuration
BLOCK_YAW_RANGE = (-np.pi / 4, np.pi / 4)  # grasp yaw tracks the block yaw

# --------------------------------------------------------------------------
# Gripper commands
#
# GRIP_OPEN is *not* the full 0.044. A fully open finger reaches 43.9 mm from the
# tool axis; a neighbouring block's near face is only 52.5-17.5 = 35 mm away at
# the minimum spawn gap, so the finger would strike it on the way down. 0.030
# gives a 60.2 mm gap: 12.6 mm clearance around a 35 mm block per side.
# --------------------------------------------------------------------------
GRIP_OPEN = 0.030
GRIP_CLOSE = 0.0  # commanded shut; the block stops the fingers (~0.7 N squeeze)
GRIP_CLOSE_TIME = 0.45  # s to hold the close command before lifting
GRIP_OPEN_TIME = 0.25  # s to hold the release command before retreating

# --------------------------------------------------------------------------
# Approach geometry (all relative to a block centre height zc)
# --------------------------------------------------------------------------
# Grasp with the TCP 7 mm above the block centre. Below ~5.5 mm the pads foul the
# mat (BLOCK_HALF + dz - GRIPPER_REACH_BELOW_TCP > MAT clearance); this leaves
# 4.5 mm and still puts 26.5 mm of pad on the block's 35 mm face.
GRASP_DZ = 0.007
PREGRASP_DZ = 0.060  # TCP above the grasp pose before descending
LIFT_DZ = 0.080  # TCP above the grasp pose after closing
HOVER_DZ = 0.045  # TCP above the place pose before descending
RELEASE_GAP = 0.002  # place the block this far above its resting height
# Every horizontal move happens at safe_transit_z(): a carried block's underside
# clears the tallest stack by at least this much.
TRANSIT_CLEARANCE = 0.030

# --------------------------------------------------------------------------
# Servo loop (mirrors engine.servo_to; tolerances tuned for episode length)
# --------------------------------------------------------------------------
GAIN = 3.0  # DiffIK pos/ori gain -- engine.py's measured value
POS_TOL = 1e-3  # m, at grasp/place
ORI_TOL = np.deg2rad(1.0)
HOLD = 0.30  # s inside tolerance before a waypoint counts as reached
POS_TOL_TRANSIT = 5e-3  # looser while moving through free space
ORI_TOL_TRANSIT = np.deg2rad(5.0)
HOLD_TRANSIT = 0.10
TIMEOUT = 8.0  # s of sim time per waypoint

# Continuous Cartesian trajectories.  Interior semantic waypoints are blended
# and passed without settling; these limits apply to the single time law over a
# complete motion phase.  Values are deliberately conservative for the small L5.
TRAJ_LINEAR_SPEED = 0.12  # m/s
TRAJ_LINEAR_ACCEL = 0.50  # m/s^2
TRAJ_LINEAR_JERK = 4.0  # m/s^3
TRAJ_ANGULAR_SPEED = 1.0  # rad/s
TRAJ_ANGULAR_ACCEL = 3.0  # rad/s^2
TRAJ_ANGULAR_JERK = 20.0  # rad/s^3
TRAJ_JOINT_ACCEL = 8.0  # rad/s^2, final guard after differential IK
TRAJ_BLEND_RADIUS = 0.012  # m; leaves at least 18 mm of transit clearance
TRAJ_MIN_DURATION = 0.25  # s

SETTLE_TIME = 1.0  # s of sim time before evaluating success predicates
STACK_SETTLE_TIME = 0.5  # s to let a pre-built stack come to rest
START_SETTLE_TIME = 0.3  # s after reset, before the episode proper begins

# --------------------------------------------------------------------------
# Success predicates (ground truth, after SETTLE_TIME)
# --------------------------------------------------------------------------
XY_TOL = 0.02  # m, block vs cross / vs the block below
Z_TOL = 0.010  # m, stacked block vs its expected height
SPEED_TOL = 0.01  # m/s, linear speed of every block
MAX_FAILURE_RATE = 0.20  # abort a task above this
MAX_ATTEMPTS_PER_EPISODE = 8

# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------
CAMERAS = ("third_person", "wrist_cam")
CAM_HEIGHT = 480
CAM_WIDTH = 640
CAM_FPS = 30.0  # nominal; recorded timestamps are authoritative
CAM_PERIOD = 1.0 / CAM_FPS

# Depth is stored as uint16 millimetres (see README): 1 mm quantisation, 0 = invalid.
# The far plane is znear/zfar * extent = 29.65 m; sky pixels come back at exactly
# zfar, so anything past 99% of it is "no geometry".
DEPTH_SCALE = 1000.0
DEPTH_INVALID = 0
DEPTH_MAX_MM = 65535

# --------------------------------------------------------------------------
# HDF5
# --------------------------------------------------------------------------
# Measured on real frames: gzip-4 gives 13.1x (0.235 MB per two-camera tick) at
# 254 MB/s in. lzf is 2x faster but only 8.0x. Storage, not time, is the binding
# constraint here, so gzip-4.
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4
PROPRIO_CHUNK = 2048  # rows
CAM_CHUNK = 1  # one frame per chunk: appends never read-modify-write

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
CATEGORY_CODE = {"a": 1, "b": 2, "c": 3}


def episode_seed(category, task_id, episode_id, attempt=0):
    """Stable seed for one episode attempt.

        seed = 100_000_000*cat + 1_000_000*task_id + episode_id + 10_000*attempt

    Task blocks are 1e6 apart and attempts step by 1e4, so with episode_id <= 240
    and attempt <= MAX_ATTEMPTS_PER_EPISODE no two (task, episode, attempt) triples
    can collide. Attempt 0 is the first try; a discarded attempt never reuses a seed.
    """
    if not 0 <= attempt <= MAX_ATTEMPTS_PER_EPISODE:
        raise ValueError(f"attempt {attempt} out of range")
    code = CATEGORY_CODE[category]
    return 100_000_000 * code + 1_000_000 * task_id + episode_id + 10_000 * attempt


def split_of(episode_id):
    """Every 10th episode of a task is validation."""
    return "val" if episode_id % 10 == 0 else "train"


def episode_filename(task_id, episode_id):
    return f"task_{task_id}_episode_{episode_id}.hdf5"


def r_max(z):
    """Conservative tool-down reach at height ``z`` (linear interp of R_MAX_BY_Z)."""
    zs = np.array([p[0] for p in R_MAX_BY_Z])
    rs = np.array([p[1] for p in R_MAX_BY_Z])
    return float(np.interp(z, zs, rs))


def reachable(pos):
    """Is this TCP position inside the verified tool-down workspace?"""
    x, y, z = pos
    return np.hypot(x, y) <= r_max(z)


def tool_down_quat(yaw=np.pi / 2):
    """World quat (wxyz) putting the TCP axis (+x of link6) along -z world.

    ``yaw`` is the world direction of the finger closing axis (+y of link6), so
    ``yaw = block_yaw + pi/2`` closes the fingers onto a cube's opposite faces.
    ``yaw = pi/2`` is the minimal rotation from the home pose (joint6 stays at 0).
    """
    import mujoco

    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    rot = np.column_stack([x, y, np.cross(x, y)])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot.ravel()))
    return quat


def grasp_yaw(block_yaw):
    """Closing-axis yaw that squares the fingers to a cube yawed by ``block_yaw``."""
    return block_yaw + np.pi / 2


def block_center_z(level):
    """Centre height of the ``level``-th block in a stack (0 = on the mat)."""
    return MAT_TOP + BLOCK_HALF + level * BLOCK_SIZE


def assert_model_matches(model):
    """Fail loudly if the compiled model drifted from what Phase 0 measured."""
    import mujoco

    if abs(model.opt.timestep - EXPECTED_TIMESTEP) > 1e-12:
        raise RuntimeError(f"timestep {model.opt.timestep} != {EXPECTED_TIMESTEP}")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    if aid == -1:
        raise RuntimeError(f"no actuator named {GRIPPER_ACTUATOR!r}")
    lo, hi = model.actuator_ctrlrange[aid]
    if (abs(lo - GRIPPER_CTRL_RANGE[0]) > 1e-9) or (abs(hi - GRIPPER_CTRL_RANGE[1]) > 1e-9):
        raise RuntimeError(f"gripper ctrlrange {(lo, hi)} != {GRIPPER_CTRL_RANGE}")
    if (model.vis.global_.offwidth, model.vis.global_.offheight) < (CAM_WIDTH, CAM_HEIGHT):
        raise RuntimeError("offscreen framebuffer smaller than the capture resolution")
    return aid
