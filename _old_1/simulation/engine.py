"""Drive the ARX L5 through a sequence of end-effector poses with differential IK.

    uv run python simulation/engine.py             # 3D viewer + camera canvas
    uv run python simulation/engine.py --headless  # no windows, just the report
    uv run python simulation/engine.py --no-canvas # viewer only

In the viewer, the target frame is an RGB triad (red/green/blue = target x/y/z)
and the small yellow sphere is the tool center point the IK is servoing.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from camera import CameraCanvas
from ik import DiffIK
from servo import CriticalDamper

SCENE = "/home/yuxin/workspace/RoboPref/simulation/assets/robots/arx_l5/scene.xml"

POS_TOL = 1e-3  # m
ORI_TOL = np.deg2rad(1.0)
HOLD = 0.5  # seconds the pose must stay inside tolerance before we call it reached
TIMEOUT = 8.0  # seconds per waypoint

# Errors decay with time constant 1/gain; 3.0 keeps peak joint speed near 1 rad/s.
GAIN = 3.0

AXIS_RGBA = ((1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.2, 0.4, 1, 1))


def waypoints(p0, q0):
    """A tour of poses defined relative to the home tool pose."""

    def turned(axis, angle):
        delta, quat = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(delta, np.array(axis, dtype=float), angle)
        mujoco.mju_mulQuat(quat, delta, q0)
        return quat

    return [
        ("reach forward-left", p0 + [0.08, 0.10, 0.00], q0),
        ("drop + yaw wrist", p0 + [0.05, 0.00, -0.10], turned([0, 0, 1], 0.6)),
        ("lift + pitch wrist", p0 + [0.00, -0.10, 0.10], turned([0, 1, 0], -0.5)),
        ("return home", p0, q0),
    ]


def _add_geom(scn, gtype, size, pos, rgba):
    """Append a geom to a user scene, or return None if it is full."""
    if scn.ngeom >= scn.maxgeom:
        return None
    scn.ngeom += 1
    geom = scn.geoms[scn.ngeom - 1]
    mujoco.mjv_initGeom(
        geom,
        gtype,
        np.asarray(size, dtype=float),
        np.ascontiguousarray(pos, dtype=float),
        np.eye(3).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    return geom


def draw_markers(scn, target_pos, target_quat, tcp_pos, axis_len=0.07):
    """Overlay the target frame and the current TCP on the viewer's user scene.

    Expects ``viewer.user_scn``, which holds only our markers -- clearing it each
    frame is safe. Do not hand this a full MjvScene; it would drop the robot.
    """
    scn.ngeom = 0

    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.ascontiguousarray(target_quat, dtype=float))
    rot = rot.reshape(3, 3)

    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], target_pos, (1, 1, 1, 0.4))
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], tcp_pos, (1, 0.9, 0.1, 1))

    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    for axis in range(3):
        geom = _add_geom(scn, arrow, np.zeros(3), np.zeros(3), AXIS_RGBA[axis])
        if geom is None:
            return
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.004,
            np.ascontiguousarray(target_pos, dtype=float),
            np.ascontiguousarray(target_pos + rot[:, axis] * axis_len, dtype=float),
        )


def servo_to(model, data, ik, damper, target_pos, target_quat, q_cmd, viewer=None, canvas=None):
    """Run the IK loop until the tool holds the target pose, or we time out.

    Returns the updated joint command, the final (position, orientation) error,
    and how far past the target the tool travelled on the way in.
    """
    settled = 0.0
    pos_err = ori_err = np.inf
    deadline = data.time + TIMEOUT
    realtime = viewer is not None or canvas is not None

    # Progress along the straight line in: 1.0 is the target, further is overshoot.
    start = ik.tool_pose(data)[0].copy()
    span = np.linalg.norm(target_pos - start)
    approach = (target_pos - start) / span if span > 1e-9 else np.zeros(3)
    furthest = 0.0

    while data.time < deadline:
        step_start = time.time()

        damper.apply(data)
        q_cmd = ik.step(data, target_pos, target_quat, q_cmd)
        data.ctrl[:6] = q_cmd
        mujoco.mj_step(model, data)

        tcp = ik.tool_pose(data)[0]
        furthest = max(furthest, (tcp - start) @ approach)

        err = ik.error(data, target_pos, target_quat)
        pos_err, ori_err = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
        settled = settled + model.opt.timestep if pos_err < POS_TOL and ori_err < ORI_TOL else 0.0

        if viewer is not None:
            if not viewer.is_running():
                break
            draw_markers(viewer.user_scn, target_pos, target_quat, tcp)
            viewer.sync()
        if canvas is not None:
            if canvas.closed:
                break
            canvas.update(data)

        if settled >= HOLD:
            break
        if realtime:
            lag = model.opt.timestep - (time.time() - step_start)
            if lag > 0:
                time.sleep(lag)

    return q_cmd, pos_err, ori_err, max(furthest - span, 0.0)


def run(headless=False, show_canvas=True):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    ik = DiffIK(model, pos_gain=GAIN, ori_gain=GAIN)
    damper = CriticalDamper(model)

    p0, q0 = ik.tool_pose(data)
    q_cmd = data.qpos[ik.qadr].copy()

    viewer = None if headless else mujoco.viewer.launch_passive(model, data)
    canvas = CameraCanvas(model) if show_canvas and not headless else None
    try:
        for name, target_pos, target_quat in waypoints(p0, q0):
            if viewer is not None and not viewer.is_running():
                break
            if canvas is not None and canvas.closed:
                break

            t0 = data.time
            q_cmd, pos_err, ori_err, overshoot = servo_to(
                model, data, ik, damper, target_pos, target_quat, q_cmd, viewer, canvas
            )

            reached = pos_err < POS_TOL and ori_err < ORI_TOL
            print(
                f"{'reached ' if reached else 'TIMEOUT '} {name:<20} "
                f"target={np.array2string(target_pos, precision=3, suppress_small=True)}  "
                f"pos_err={pos_err * 1e3:6.3f} mm  ori_err={np.rad2deg(ori_err):6.3f} deg  "
                f"overshoot={overshoot * 1e3:7.3f} mm  ({data.time - t0:.2f}s)"
            )
    finally:
        if viewer is not None:
            viewer.close()
        if canvas is not None:
            canvas.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="no windows, just the report")
    parser.add_argument("--no-canvas", action="store_true", help="skip the camera canvas")
    args = parser.parse_args()
    run(headless=args.headless, show_canvas=not args.no_canvas)
