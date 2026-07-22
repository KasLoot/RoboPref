"""Synchronized HDF5 episode recorder.

Proprio and actions land every physics step; cameras whenever sim time crosses the
next 1/30 s tick. Recorded timestamps are authoritative -- the camera rate is
nominal, never exact, because a 2 ms physics step does not divide 1/30 s.

Everything streams: a 30 s episode holds ~850 camera ticks, which is 2.6 GB of raw
pixels, far too much to buffer. Frames are appended to resizable datasets as they
are captured. The file is written to ``<name>.partial`` and renamed only once the
episode is known to have succeeded, so a discarded attempt is never left behind.

Depth is stored as uint16 millimetres rather than float32: it halves the depth
bytes, 1 mm quantisation is well below this scene's geometric fidelity, and the
whole 0..65.5 m range fits. 0 means "no geometry" (sky, or past the far plane).
"""

import os

import h5py
import mujoco
import numpy as np

import config as C


def depth_to_mm(depth_m, zfar):
    """Metres (float32) -> millimetres (uint16). 0 = invalid.

    The renderer returns the far plane distance for pixels with no geometry, so
    anything at 99% of zfar or beyond is background, not a measurement.
    """
    invalid = (depth_m <= 0.0) | (depth_m >= 0.99 * zfar)
    mm = np.clip(depth_m * C.DEPTH_SCALE, 1.0, C.DEPTH_MAX_MM)
    return np.where(invalid, C.DEPTH_INVALID, mm).astype(np.uint16)


def camera_params(model, data, name):
    """Intrinsics and episode-start extrinsics for one camera."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid == -1:
        raise ValueError(f"no camera {name!r}")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(data.cam_xmat[cid].ravel()))
    return {
        "fovy": float(model.cam_fovy[cid]),
        "resolution": np.array([C.CAM_WIDTH, C.CAM_HEIGHT], dtype=np.int32),
        "native_resolution": model.cam_resolution[cid].astype(np.int32),
        "sensorsize": model.cam_sensorsize[cid].astype(np.float64),
        "focal": model.cam_intrinsic[cid][:2].astype(np.float64),
        "principal": model.cam_intrinsic[cid][2:].astype(np.float64),
        "pos": data.cam_xpos[cid].astype(np.float64),
        "quat_wxyz": quat,
        "body_mounted": bool(model.cam_bodyid[cid] != 0),
    }


class EpisodeRecorder:
    """One episode -> one HDF5 file. Use as a context manager; call ``keep()`` to
    commit, otherwise the partial file is removed on exit."""

    def __init__(self, path, scene, data, attrs):
        self.path = str(path)
        self.tmp = self.path + ".partial"
        self.scene = scene
        self.model = scene.model
        self.attrs = dict(attrs)
        self._kept = False

        self.zfar = float(self.model.vis.map.zfar * self.model.stat.extent)
        self._next_cam_t = 0.0
        self._buf = {k: [] for k in
                     ("t", "joint_pos", "joint_vel", "joint_torque", "ee_pos",
                      "ee_quat_wxyz", "q_cmd", "gripper")}
        self.n_steps = 0
        self.n_frames = 0

        self._renderer = mujoco.Renderer(self.model, C.CAM_HEIGHT, C.CAM_WIDTH)
        self._f = h5py.File(self.tmp, "w")
        self._make_datasets(data)

    # -- setup --------------------------------------------------------------
    def _ds(self, name, shape, dtype, chunk):
        return self._f.create_dataset(
            name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
            chunks=(chunk, *shape), compression=C.COMPRESSION,
            compression_opts=C.COMPRESSION_OPTS)

    def _make_datasets(self, data):
        n = C.PROPRIO_CHUNK
        self._ds("/proprio/t", (), "f8", n)
        self._ds("/proprio/joint_pos", (6,), "f4", n)
        self._ds("/proprio/joint_vel", (6,), "f4", n)
        self._ds("/proprio/joint_torque", (6,), "f4", n)
        self._ds("/proprio/ee_pos", (3,), "f4", n)
        self._ds("/proprio/ee_quat_wxyz", (4,), "f4", n)
        self._ds("/actions/t", (), "f8", n)
        self._ds("/actions/q_cmd", (6,), "f4", n)
        self._ds("/actions/gripper", (), "f4", n)

        for cam in C.CAMERAS:
            self._ds(f"/cameras/{cam}/t", (), "f8", C.PROPRIO_CHUNK)
            self._ds(f"/cameras/{cam}/rgb", (C.CAM_HEIGHT, C.CAM_WIDTH, 3), "u1", C.CAM_CHUNK)
            self._ds(f"/cameras/{cam}/depth", (C.CAM_HEIGHT, C.CAM_WIDTH), "u2", C.CAM_CHUNK)

            params = camera_params(self.model, data, cam)
            group = self._f[f"/cameras/{cam}"]
            for key, value in params.items():
                group.attrs[key] = value
                self.attrs[f"camera_{cam}_{key}"] = value

    # -- streaming ----------------------------------------------------------
    def on_step(self, t, qpos, qvel, qtau, ee_pos, ee_quat, q_cmd, grip, data):
        b = self._buf
        b["t"].append(t)
        b["joint_pos"].append(qpos)
        b["joint_vel"].append(qvel)
        b["joint_torque"].append(qtau)
        b["ee_pos"].append(ee_pos)
        b["ee_quat_wxyz"].append(ee_quat)
        b["q_cmd"].append(q_cmd)
        b["gripper"].append(grip)
        self.n_steps += 1
        if len(b["t"]) >= C.PROPRIO_CHUNK:
            self._flush_proprio()

        if t + 1e-9 >= self._next_cam_t:
            self._next_cam_t += C.CAM_PERIOD
            self._capture(t, data)

    def _capture(self, t, data):
        for cam in C.CAMERAS:
            self._renderer.update_scene(data, camera=cam)
            self._renderer.disable_depth_rendering()
            rgb = self._renderer.render()
            self._renderer.enable_depth_rendering()
            depth = depth_to_mm(self._renderer.render(), self.zfar)

            g = self._f[f"/cameras/{cam}"]
            k = g["t"].shape[0]
            for name, value in (("t", t), ("rgb", rgb), ("depth", depth)):
                g[name].resize(k + 1, axis=0)
                g[name][k] = value
        self._renderer.disable_depth_rendering()
        self.n_frames += 1

    def _append(self, name, values):
        ds = self._f[name]
        k = ds.shape[0]
        ds.resize(k + len(values), axis=0)
        ds[k:] = np.asarray(values)

    def _flush_proprio(self):
        b = self._buf
        if not b["t"]:
            return
        self._append("/proprio/t", b["t"])
        self._append("/proprio/joint_pos", b["joint_pos"])
        self._append("/proprio/joint_vel", b["joint_vel"])
        self._append("/proprio/joint_torque", b["joint_torque"])
        self._append("/proprio/ee_pos", b["ee_pos"])
        self._append("/proprio/ee_quat_wxyz", b["ee_quat_wxyz"])
        self._append("/actions/t", b["t"])
        self._append("/actions/q_cmd", b["q_cmd"])
        self._append("/actions/gripper", b["gripper"])
        for v in b.values():
            v.clear()

    # -- teardown -----------------------------------------------------------
    @property
    def kept(self):
        return self._kept

    def keep(self, extra_attrs=None):
        """Commit: flush, stamp attrs, rename into place."""
        self._flush_proprio()
        attrs = dict(self.attrs)
        attrs.update(extra_attrs or {})
        attrs["success"] = True
        attrs["n_steps"] = self.n_steps
        attrs["n_camera_frames"] = self.n_frames
        for key, value in attrs.items():
            self._f.attrs[key] = value
        self._kept = True

    def close(self):
        self._renderer.close()
        self._f.close()
        if self._kept:
            os.replace(self.tmp, self.path)
        elif os.path.exists(self.tmp):
            os.remove(self.tmp)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
def integrity_check(path, deep=False):
    """Cheap check used by ``--resume`` to decide whether an episode is done.

    Returns ``(ok, reason)``. ``deep=True`` also verifies the row counts line up.
    """
    try:
        with h5py.File(path, "r") as f:
            if not f.attrs.get("success", False):
                return False, "success attr missing/false"
            if f.attrs.get("schema_version") != C.SCHEMA_VERSION:
                return False, "schema version mismatch"
            need = ["/proprio/t", "/proprio/joint_pos", "/proprio/joint_vel",
                    "/proprio/joint_torque", "/proprio/ee_pos", "/proprio/ee_quat_wxyz",
                    "/actions/t", "/actions/q_cmd", "/actions/gripper"]
            need += [f"/cameras/{c}/{d}" for c in C.CAMERAS for d in ("t", "rgb", "depth")]
            for key in need:
                if key not in f:
                    return False, f"missing {key}"
            n = f["/proprio/t"].shape[0]
            if n == 0:
                return False, "empty episode"
            if int(f.attrs.get("n_steps", -1)) != n:
                return False, "n_steps attr disagrees with /proprio/t"
            for cam in C.CAMERAS:
                k = f[f"/cameras/{cam}/t"].shape[0]
                if k == 0 or f[f"/cameras/{cam}/rgb"].shape[0] != k:
                    return False, f"camera {cam} frame count mismatch"
            if deep:
                for key in need:
                    if f[key].shape[0] not in (n, f[f"/cameras/{C.CAMERAS[0]}/t"].shape[0]):
                        return False, f"{key} row count {f[key].shape[0]}"
        return True, "ok"
    except (OSError, KeyError) as exc:
        return False, f"unreadable: {exc}"
