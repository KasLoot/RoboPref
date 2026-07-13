"""Per-episode task scenes, built with MjSpec and recompiled from scratch.

The robot definition is never touched: we load ``scene.xml`` into an ``MjSpec``,
append the mats, the cross marker and the free-joint blocks, rewrite the ``home``
keyframe to cover the new degrees of freedom, and compile.

Rewriting the keyframe is not optional. MuJoCo happily compiles a model whose
``key_qpos`` is shorter than ``nq`` -- it zero-pads -- so a stale keyframe would
silently teleport every block to the world origin on ``mj_resetDataKeyframe``.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

import config as C


@dataclass(frozen=True)
class BlockPlacement:
    """Where one block starts. ``z`` is the centre height."""

    color: str
    x: float
    y: float
    z: float
    yaw: float = 0.0

    @property
    def xy(self):
        return np.array([self.x, self.y])

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def quat(self):
        return np.array([np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)])


@dataclass
class EpisodeScene:
    """A compiled model plus the ground truth needed to script and score it."""

    model: mujoco.MjModel
    placements: list
    cross_xy: np.ndarray
    init_stack: list  # bottom -> top colors, empty if none
    mat_colors: list  # colors loose on the black mat
    gripper_act: int
    body_id: dict = field(default_factory=dict)
    qpos_adr: dict = field(default_factory=dict)
    qvel_adr: dict = field(default_factory=dict)

    def block_pos(self, data, color):
        return data.xpos[self.body_id[color]].copy()

    def block_quat(self, data, color):
        return data.xquat[self.body_id[color]].copy()

    def block_speed(self, data, color):
        adr = self.qvel_adr[color]
        return float(np.linalg.norm(data.qvel[adr:adr + 3]))

    def block_yaw(self, data, color):
        w, x, y, z = self.block_quat(data, color)
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_cross(rng):
    """Cross centre, uniform in the centred CROSS_REGION square of the white mat."""
    half = C.CROSS_REGION / 2
    return np.array(C.WHITE_MAT_CENTER) + rng.uniform(-half, half, size=2)


def sample_mat_blocks(rng, colors, z=None, forbid=()):
    """Non-overlapping poses for ``colors`` on the black mat.

    Centres stay ``SPAWN_MIN_GAP`` apart and ``SPAWN_MARGIN`` inside the mat edge.
    Placing blocks one at a time can strand the last one, so the whole configuration
    restarts rather than the sampler relaxing the constraint. If even that stalls we
    raise, and the caller resamples the episode with a derived seed.
    """
    z = C.block_center_z(0) if z is None else z
    centre = np.array(C.BLACK_MAT_CENTER)
    half = np.array(C.SPAWN_HALF)
    fixed = [np.asarray(p, dtype=float) for p in forbid]

    for _ in range(C.SPAWN_MAX_RESTARTS):
        chosen, out = list(fixed), []
        for color in colors:
            for _ in range(C.SPAWN_MAX_TRIES):
                xy = centre + rng.uniform(-1, 1, size=2) * half
                if all(np.linalg.norm(xy - o) >= C.SPAWN_MIN_GAP for o in chosen):
                    chosen.append(xy)
                    out.append(BlockPlacement(color, xy[0], xy[1], z,
                                              rng.uniform(*C.BLOCK_YAW_RANGE)))
                    break
            else:
                break  # stranded: restart the whole configuration
        if len(out) == len(colors):
            return out
    raise RuntimeError(f"could not place {list(colors)} on the black mat "
                       f"({C.SPAWN_MAX_RESTARTS} restarts)")


def stack_placements(cross_xy, colors):
    """A pre-built stack at the cross, bottom -> top. Yaw 0: a yawed stack topples."""
    return [BlockPlacement(c, cross_xy[0], cross_xy[1], C.block_center_z(i), 0.0)
            for i, c in enumerate(colors)]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def look_at_xyaxes(pos, target):
    """MuJoCo camera ``xyaxes`` (right, up) aiming ``pos`` at ``target``.

    MuJoCo looks along ``-z_cam``. With the camera on the y = 0 plane the right axis
    comes out as exactly world +y, which is what keeps "white mat = image left" true.
    """
    pos, target = np.asarray(pos, dtype=float), np.asarray(target, dtype=float)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    z = -forward
    x = np.cross([0.0, 0.0, 1.0], z)
    norm = np.linalg.norm(x)
    if norm < 1e-9:  # looking straight down: "right" is undefined
        raise ValueError(f"camera at {pos} is directly above its aim point {target}")
    return [*(x / norm), *np.cross(z, x / norm)]


def _restyle(spec):
    """Recolour the robot, re-place the third_person camera and add the
    near-top-down camera on the spec.

    Done here rather than in the XML so ``assets/robots/arx_l5`` stays pristine.
    """
    material = next(m for m in spec.materials if m.name == C.ROBOT_MATERIAL)
    material.rgba = list(C.ROBOT_RGBA)

    cam = next(c for c in spec.cameras if c.name == "third_person")
    cam.pos = list(C.THIRD_PERSON_POS)
    cam.alt.xyaxes = look_at_xyaxes(C.THIRD_PERSON_POS, C.THIRD_PERSON_AIM)

    top = spec.worldbody.add_camera(name="top_cam", pos=list(C.TOP_CAM_POS))
    top.alt.xyaxes = look_at_xyaxes(C.TOP_CAM_POS, C.TOP_CAM_AIM)


def _add_mat(spec, name, centre, rgba):
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[C.MAT_HALF[0], C.MAT_HALF[1], C.MAT_THICK / 2],
        pos=[centre[0], centre[1], C.MAT_THICK / 2],
        rgba=list(rgba),
        condim=C.BLOCK_CONDIM,
        friction=list(C.BLOCK_FRICTION),
    )


def _add_cross(spec, cross_xy):
    """Two thin bright bars flush on the white mat. Visual only: they must not
    perturb a block placed on top of them."""
    z = C.MAT_TOP + C.CROSS_THICK / 2
    bars = {
        "cross_x": [C.CROSS_ARM_HALF, C.CROSS_ARM_WIDTH, C.CROSS_THICK / 2],
        "cross_y": [C.CROSS_ARM_WIDTH, C.CROSS_ARM_HALF, C.CROSS_THICK / 2],
    }
    for name, size in bars.items():
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=[cross_xy[0], cross_xy[1], z],
            rgba=list(C.CROSS_RGBA),
            contype=0,
            conaffinity=0,
        )


def build_scene(placements, cross_xy, init_stack=(), mat_colors=()):
    """Compile a fresh model with these blocks. Returns an :class:`EpisodeScene`."""
    spec = mujoco.MjSpec.from_file(C.SCENE_XML)

    key = spec.keys[0]
    if key.name != "home":
        raise RuntimeError(f"expected keyframe 'home', found {key.name!r}")
    arm_qpos = np.asarray(key.qpos, dtype=float)[:6].copy()
    arm_ctrl = np.asarray(key.ctrl, dtype=float)[:6].copy()

    _restyle(spec)
    _add_mat(spec, "white_mat", C.WHITE_MAT_CENTER, C.WHITE_MAT_RGBA)
    _add_mat(spec, "black_mat", C.BLACK_MAT_CENTER, C.BLACK_MAT_RGBA)
    _add_cross(spec, cross_xy)

    for p in placements:
        body = spec.worldbody.add_body(name=f"block_{p.color}", pos=list(p.pos), quat=list(p.quat))
        body.add_freejoint(name=f"block_{p.color}_free")
        body.add_geom(
            name=f"block_{p.color}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[C.BLOCK_HALF] * 3,
            rgba=list(C.COLOR_RGBA[p.color]),
            mass=C.BLOCK_MASS,
            condim=C.BLOCK_CONDIM,
            friction=list(C.BLOCK_FRICTION),
        )

    # The keyframe must now describe every dof, blocks included.
    qpos = [*arm_qpos, C.GRIP_OPEN, -C.GRIP_OPEN]
    for p in placements:
        qpos.extend([*p.pos, *p.quat])
    key.qpos = np.array(qpos, dtype=float)
    key.ctrl = np.array([*arm_ctrl, C.GRIP_OPEN], dtype=float)

    model = spec.compile()
    gripper_act = C.assert_model_matches(model)
    if len(key.qpos) != model.nq:
        raise RuntimeError(f"keyframe qpos {len(key.qpos)} != nq {model.nq}")

    scene = EpisodeScene(
        model=model,
        placements=list(placements),
        cross_xy=np.asarray(cross_xy, dtype=float),
        init_stack=list(init_stack),
        mat_colors=list(mat_colors),
        gripper_act=gripper_act,
    )
    for p in placements:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"block_{p.color}")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"block_{p.color}_free")
        scene.body_id[p.color] = bid
        scene.qpos_adr[p.color] = model.jnt_qposadr[jid]
        scene.qvel_adr[p.color] = model.jnt_dofadr[jid]
    return scene


def reset(scene):
    """Fresh MjData at the ``home`` keyframe, blocks at their spawn poses."""
    data = mujoco.MjData(scene.model)
    mujoco.mj_resetDataKeyframe(scene.model, data, 0)
    mujoco.mj_forward(scene.model, data)
    return data


def settle(scene, data, seconds):
    """Step physics holding the keyframe ctrl. No damper: the arm is not moving."""
    for _ in range(int(round(seconds / scene.model.opt.timestep))):
        mujoco.mj_step(scene.model, data)


def stack_is_intact(scene, data, tol_xy=0.008, tol_z=0.006, tol_speed=0.01):
    """Did the pre-built stack survive its settle, upright and on the cross?"""
    for level, color in enumerate(scene.init_stack):
        pos = scene.block_pos(data, color)
        if np.linalg.norm(pos[:2] - scene.cross_xy) > tol_xy:
            return False
        if abs(pos[2] - C.block_center_z(level)) > tol_z:
            return False
        if scene.block_speed(data, color) > tol_speed:
            return False
        # upright: the block's local z must still point up
        rot = data.xmat[scene.body_id[color]].reshape(3, 3)
        if rot[2, 2] < np.cos(np.deg2rad(10)):
            return False
    return True
