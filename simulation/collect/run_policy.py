"""Closed-loop pi05 policy evaluation in the RoboPref simulation.

The policy sees exactly what training saw: the same 640x480 renders of the same
two cameras, the same state convention (measured arm joints + last *commanded*
gripper), the same instruction strings, and the same norm stats (loaded from the
checkpoint's ``assets/arx_l5/``). Its (50, 7) chunk of actuator setpoints is
played through the sim's canonical 500 Hz loop -- ``CriticalDamper.apply`` every
step, no IK -- holding each 50 Hz row for 10 physics steps.

Two modes:

    # Oracle replay (no policy): a recorded episode's actions through the exact
    # execution path. Must succeed before any policy result is trusted.
    uv run python simulation/collect/run_policy.py \
        --replay /data/datasets/RoboPref_dataset/stacking_blocks_ordered/task_1_episode_5.hdf5

    # Benchmark a fine-tuned checkpoint on fresh, provably unseen seeds.
    uv run python simulation/collect/run_policy.py --dataset ordered \
        --checkpoint /data/models/pi05_arx_ordered --episodes 20 [--video] [--viewer]

Evaluation seeds: episode ids start at 301. Collected data used ids 1-240 with
retry seeds offset by 10 000 per attempt, so ids 301-999 collide with nothing
that was ever recorded (and keep the episode-id-derived task knobs varied).

Success is category-aware: B and C use the task's own predicate; category A --
whose instruction never names a stacking order -- accepts *any* completed stack
on the cross (the recorded predicate would cap a correct policy near 1/6).
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

# mujoco.viewer needs a GLFW context, which cannot share a thread with an EGL
# offscreen renderer -- same trick as collect.py's --preview.
if "--viewer" in sys.argv:
    os.environ["MUJOCO_GL"] = "glfw"

import config as C  # noqa: E402  (must precede mujoco: it sets MUJOCO_GL)

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import expert as E  # noqa: E402
import scene_builder as SB  # noqa: E402
from ik import ARM_JOINTS  # noqa: E402
from servo import CriticalDamper  # noqa: E402
from tasks import CATEGORIES  # noqa: E402

if str(C.REPO_DIR) not in sys.path:
    sys.path.insert(0, str(C.REPO_DIR))

from pi05.embodiment import ARX_L5, chunk_indices  # noqa: E402

DATASETS = {"ambiguous": "a", "ordered": "b", "decomposed": "c"}
STRIDE = ARX_L5.stride  # physics steps per chunk row (500 Hz / 50 Hz)
HORIZON = ARX_L5.horizon
EVAL_EPISODE_START = 301
VIDEO_FPS = 10.0
VIDEO_CAMERA = ARX_L5.image_obs_keys["base_0_rgb"]


def arm_qpos_addresses(model) -> np.ndarray:
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS])
    return model.jnt_qposadr[ids]


# --------------------------------------------------------------------------- #
# Success predicates                                                           #
# --------------------------------------------------------------------------- #
def verify_any_stack(scene, data, blocks):
    """Is *some* complete stack of ``blocks`` standing on the cross?"""
    reasons = []
    for perm in itertools.permutations(blocks):
        ok, reason = E.verify_stack(scene, data, list(perm))
        if ok:
            return True, f"ok ({'-'.join(perm)})"
        reasons.append(reason)
    return False, reasons[0]  # first permutation's reason as a representative


def episode_verifier(category, plan):
    if category == "a":
        return lambda scene, data: verify_any_stack(scene, data, plan.blocks_present)
    return plan.verify


# --------------------------------------------------------------------------- #
# Action execution                                                             #
# --------------------------------------------------------------------------- #
class ChunkExecutor:
    """Plays 50 Hz action rows through the canonical 500 Hz servo loop.

    Arm targets are linearly interpolated from the previous row over the 10
    substeps (the demos' q_cmd is 500 Hz-smooth, so this restores what the
    downsampling removed); ``--interp zoh`` holds each row instead. The gripper
    is a step signal and is always held. ``prev_q`` starts at the measured home
    pose and the gripper at GRIP_OPEN, exactly like the scripted expert.
    """

    def __init__(self, scene, data, interp="linear", on_substep=None):
        assert interp in ("linear", "zoh")
        self.scene = scene
        self.data = data
        self.model = scene.model
        self.interp = interp
        self.on_substep = on_substep
        self.damper = CriticalDamper(self.model)
        self.qadr = arm_qpos_addresses(self.model)
        self.prev_q = data.qpos[self.qadr].copy()
        self.grip = float(C.GRIP_OPEN)

    def _substep(self, q_target, grip):
        self.damper.apply(self.data)
        self.data.ctrl[:6] = q_target
        self.data.ctrl[self.scene.gripper_act] = grip
        mujoco.mj_step(self.model, self.data)
        if self.on_substep is not None:
            self.on_substep(self.data)

    def execute(self, chunk, n_rows):
        """Run the first ``n_rows`` of a (>=n_rows, 7) chunk."""
        for row in np.asarray(chunk, dtype=float)[:n_rows]:
            q6, grip = row[:6], float(row[6])
            for s in range(STRIDE):
                if self.interp == "linear":
                    q = self.prev_q + (s + 1) / STRIDE * (q6 - self.prev_q)
                else:
                    q = q6
                self._substep(q, grip)
            self.prev_q = q6.copy()
            self.grip = grip

    def run_tape(self, actions):
        """Raw 500 Hz replay: one recorded (7,) row per physics step."""
        for row in np.asarray(actions, dtype=float):
            self._substep(row[:6], float(row[6]))
        self.prev_q = np.asarray(actions[-1, :6], dtype=float).copy()
        self.grip = float(actions[-1, 6])

    def hold(self, seconds):
        """Freeze the command and let physics settle (mirrors Expert.hold)."""
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            self._substep(self.prev_q, self.grip)


# --------------------------------------------------------------------------- #
# Episode setup / rollout                                                      #
# --------------------------------------------------------------------------- #
def build_episode(module, task_id, episode_id, seed=None):
    """Scene + fresh MjData, mirroring collect.run_episode: settled (and intact)
    init stack, episode clock zeroed. ``seed=None`` derives per-attempt seeds."""
    for attempt in range(C.MAX_ATTEMPTS_PER_EPISODE + 1):
        s = seed if seed is not None else C.episode_seed(module.CATEGORY, task_id, episode_id, attempt)
        plan = module.build(task_id, episode_id, s)
        data = SB.reset(plan.scene)
        if plan.init_stack:
            SB.settle(plan.scene, data, C.STACK_SETTLE_TIME)
            if not SB.stack_is_intact(plan.scene, data):
                if seed is not None:
                    raise RuntimeError(f"recorded seed {seed}: pre-built stack collapsed")
                continue
        data.time = 0.0
        return plan, data, s
    raise RuntimeError(f"no intact scene for task {task_id} episode {episode_id}")


def run_policy_episode(category, plan, data, policy_fn, *, execute_horizon, time_limit,
                       interp, snap_gripper=True, early_stop=True, on_substep=None):
    scene = plan.scene
    renderer = mujoco.Renderer(scene.model, C.CAM_HEIGHT, C.CAM_WIDTH)
    executor = ChunkExecutor(scene, data, interp=interp, on_substep=on_substep)
    verify = episode_verifier(category, plan)
    replans = 0
    started = time.time()
    try:
        while data.time < time_limit:
            images = {}
            for model_key, camera in ARX_L5.image_obs_keys.items():
                renderer.update_scene(data, camera=camera)
                images[model_key] = renderer.render()
            state = np.concatenate([data.qpos[executor.qadr], [executor.grip]]).astype(np.float32)

            chunk = policy_fn(images, state, plan.instruction)[:, : ARX_L5.action_dim].copy()
            if snap_gripper:
                # Demos contain only {closed, open}; anything else is out of
                # distribution for the plant.
                half = (C.GRIP_CLOSE + C.GRIP_OPEN) / 2
                chunk[:, 6] = np.where(chunk[:, 6] > half, C.GRIP_OPEN, C.GRIP_CLOSE)
            else:
                chunk[:, 6] = np.clip(chunk[:, 6], *C.GRIPPER_CTRL_RANGE)

            executor.execute(chunk, execute_horizon)
            replans += 1
            if early_stop:
                ok, _ = verify(scene, data)
                if ok:
                    break
    finally:
        renderer.close()

    executor.hold(C.SETTLE_TIME)  # the final verdict is always post-settle
    ok, reason = verify(scene, data)
    return {"success": bool(ok), "reason": reason, "sim_time": round(data.time, 2),
            "wall_s": round(time.time() - started, 1), "replans": replans}


def replay_episode(path, *, execute_horizon, interp, raw_500hz=False, sink_factory=None):
    """Oracle sanity: the recorded actions through the exact execution path.

    Chunks are re-sliced from the recording at every replan boundary with the
    same stride-10 downsampling as training (``chunk_indices``), so this
    isolates the executor/downsampling plumbing from policy quality.

    ``sink_factory(plan, data) -> (on_substep, close_fn)`` attaches an optional
    viewer/video sink to the episode's own scene and data.
    """
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        category = str(attrs["category"])
        task_id, episode_id, seed = int(attrs["task_id"]), int(attrs["episode_id"]), int(attrs["seed"])
        actions = np.concatenate([np.asarray(f["actions/q_cmd"]),
                                  np.asarray(f["actions/gripper"])[:, None]], axis=1)

    plan, data, _ = build_episode(CATEGORIES[category], task_id, episode_id, seed=seed)
    on_substep, close_sinks = (None, None) if sink_factory is None else sink_factory(plan, data)
    executor = ChunkExecutor(plan.scene, data, interp=interp, on_substep=on_substep)
    started = time.time()
    try:
        if raw_500hz:
            executor.run_tape(actions)
        else:
            j, n = 0, len(actions)
            while j < n:
                executor.execute(actions[chunk_indices(j, n, STRIDE, HORIZON)], execute_horizon)
                j += execute_horizon * STRIDE
        executor.hold(C.SETTLE_TIME)
    finally:
        if close_sinks is not None:
            close_sinks()
    ok, reason = plan.verify(plan.scene, data)  # the demo's own (exact) predicate
    return {"file": Path(path).name, "mode": "500hz" if raw_500hz else f"{execute_horizon}rows/{interp}",
            "success": bool(ok), "reason": reason, "sim_time": round(data.time, 2),
            "wall_s": round(time.time() - started, 1)}


# --------------------------------------------------------------------------- #
# Optional sinks: live viewer, video                                           #
# --------------------------------------------------------------------------- #
class LiveViewer:
    """Passive viewer paced to wall clock. Requires --viewer (GLFW mode)."""

    def __init__(self, model, data):
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.timestep = model.opt.timestep
        self._last = time.time()

    def on_substep(self, data):
        if not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        self.viewer.sync()
        lag = self.timestep - (time.time() - self._last)
        if lag > 0:
            time.sleep(lag)
        self._last = time.time()

    def close(self):
        self.viewer.close()


class VideoSink:
    """Streams base-camera frames to MP4 via imageio, or an animated GIF via
    PIL (at half resolution) when imageio is not installed."""

    def __init__(self, path, model):
        self.renderer = mujoco.Renderer(model, C.CAM_HEIGHT, C.CAM_WIDTH)
        self.every = int(round(1.0 / (VIDEO_FPS * model.opt.timestep)))
        self._count = 0
        self._frames = []
        self.writer = None
        try:
            import imageio.v2 as imageio

            self.path = Path(path).with_suffix(".mp4")
            self.writer = imageio.get_writer(str(self.path), fps=VIDEO_FPS)
        except ImportError:
            self.path = Path(path).with_suffix(".gif")

    def on_substep(self, data):
        self._count += 1
        if self._count % self.every:
            return
        self.renderer.update_scene(data, camera=VIDEO_CAMERA)
        frame = self.renderer.render()
        if self.writer is not None:
            self.writer.append_data(frame)
        else:
            from PIL import Image

            self._frames.append(Image.fromarray(frame).reduce(2))

    def close(self):
        self.renderer.close()
        if self.writer is not None:
            self.writer.close()
        elif self._frames:
            self._frames[0].save(self.path, save_all=True, append_images=self._frames[1:],
                                 duration=int(1000 / VIDEO_FPS), loop=0)


def attach_sinks(plan, data, *, video_path=None, viewer=False):
    """Attach the optional viewer/video sinks to one episode's scene and data.

    Returns ``(on_substep hook or None, close())`` -- the single construction
    path shared by the replay and policy modes.
    """
    sinks = []
    if video_path is not None:
        sinks.append(VideoSink(video_path, plan.scene.model))
    if viewer:
        sinks.append(LiveViewer(plan.scene.model, data))

    def close():
        for sink in sinks:
            sink.close()

    if not sinks:
        return None, close

    def hook(step_data):
        for sink in sinks:
            sink.on_substep(step_data)
    return hook, close


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def make_policy(checkpoint_dir, num_steps):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "model.pt").exists():
        sys.exit(f"no model.pt under {checkpoint_dir} -- for --use-best, best/ only exists "
                 f"once training has run an action-MAE validation (every --action-val-every "
                 f"steps); use the checkpoint root for the latest weights")

    from pi05.inference import Pi05Policy  # imports torch: only in policy mode

    policy = Pi05Policy(str(checkpoint_dir), embodiment=ARX_L5.name,
                        token_len=ARX_L5.token_len)

    def policy_fn(images, state, prompt):
        return policy.infer_raw(images, state, prompt, num_steps=num_steps)
    return policy_fn


def print_table(rows, by):
    groups = {}
    for row in rows:
        groups.setdefault(row[by], []).append(row["success"])
    print(f"\n{'group':<28} {'success':>10} {'rate':>8}")
    print("-" * 48)
    for key in sorted(groups):
        outcomes = groups[key]
        print(f"{str(key):<28} {sum(outcomes):>6}/{len(outcomes):<3} {np.mean(outcomes):>7.1%}")
    total = [row["success"] for row in rows]
    print("-" * 48)
    print(f"{'overall':<28} {sum(total):>6}/{len(total):<3} {np.mean(total):>7.1%}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replay", nargs="+", type=Path, default=None,
                   help="recorded episode(s) to replay through the executor (no policy)")
    p.add_argument("--replay-500hz", action="store_true", help="replay at raw 500 Hz (upper bound)")
    p.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    p.add_argument("--checkpoint", default=None,
                   help="fine-tuned checkpoint dir (default: /data/models/pi05_arx_<dataset>)")
    p.add_argument("--use-best", action="store_true", help="load <checkpoint>/best")
    p.add_argument("--task", type=int, default=None, help="single task id (default: all)")
    p.add_argument("--episodes", type=int, default=20, help="episodes per task")
    p.add_argument("--episode-start", type=int, default=EVAL_EPISODE_START)
    p.add_argument("--execute-horizon", type=int, default=25,
                   help="chunk rows executed before replanning (25 = 0.5 s)")
    p.add_argument("--num-steps", type=int, default=10, help="flow integration steps")
    p.add_argument("--interp", choices=("linear", "zoh"), default="linear")
    p.add_argument("--time-limit", type=float, default=None,
                   help="sim seconds per episode (default: 60 stacking / 30 decomposed)")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--no-snap-gripper", action="store_true")
    p.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (wall-clock paced)")
    p.add_argument("--video", action="store_true", help="save per-episode third_person videos")
    p.add_argument("--out", type=Path, default=None, help="results dir (default: <checkpoint>/eval)")
    args = p.parse_args()

    # ---- oracle replay mode ---------------------------------------------- #
    if args.replay is not None:
        def sink_factory_for(path):
            if not (args.viewer or args.video):
                return None
            video_path = Path(path).with_name(Path(path).stem + "_replay") if args.video else None
            return lambda plan, data: attach_sinks(plan, data, video_path=video_path,
                                                   viewer=args.viewer)

        rows = []
        for path in args.replay:
            row = replay_episode(path, execute_horizon=args.execute_horizon,
                                 interp=args.interp, raw_500hz=args.replay_500hz,
                                 sink_factory=sink_factory_for(path))
            rows.append(row)
            print(json.dumps(row))
        print_table(rows, by="mode")
        return

    # ---- policy benchmark mode ------------------------------------------- #
    if args.dataset is None:
        p.error("--dataset (or --replay) is required")
    category = DATASETS[args.dataset]
    module = CATEGORIES[category]
    checkpoint = Path(args.checkpoint or f"/data/models/pi05_arx_{args.dataset}")
    if args.use_best:
        checkpoint = checkpoint / "best"
    out = args.out or (checkpoint / "eval")
    out.mkdir(parents=True, exist_ok=True)
    time_limit = args.time_limit or (30.0 if category == "c" else 60.0)
    tasks = [args.task] if args.task is not None else sorted(module.BUDGET)

    print(f"loading policy from {checkpoint} ...")
    policy_fn = make_policy(checkpoint, args.num_steps)

    results_path = out / "eval.jsonl"
    rows = []
    with results_path.open("a") as results_file:
        for task_id in tasks:
            for i in range(args.episodes):
                episode_id = args.episode_start + i
                plan, data, seed = build_episode(module, task_id, episode_id)
                video_path = None
                if args.video:
                    videos_dir = out / "videos"
                    videos_dir.mkdir(exist_ok=True)
                    video_path = videos_dir / f"task_{task_id}_episode_{episode_id}"
                hook, close_sinks = attach_sinks(plan, data, video_path=video_path,
                                                 viewer=args.viewer)
                try:
                    row = run_policy_episode(
                        category, plan, data, policy_fn,
                        execute_horizon=args.execute_horizon, time_limit=time_limit,
                        interp=args.interp, snap_gripper=not args.no_snap_gripper,
                        early_stop=not args.no_early_stop, on_substep=hook)
                finally:
                    close_sinks()
                row.update({"dataset": args.dataset, "task": task_id, "episode": episode_id,
                            "seed": seed, "instruction": plan.instruction,
                            "checkpoint": str(checkpoint),
                            "execute_horizon": args.execute_horizon, "interp": args.interp,
                            "num_steps": args.num_steps,
                            "snap_gripper": not args.no_snap_gripper})
                results_file.write(json.dumps(row) + "\n")
                results_file.flush()
                rows.append(row)
                status = "OK " if row["success"] else "FAIL"
                print(f"[{status}] task {task_id} ep {episode_id}: {row['reason']} "
                      f"(sim {row['sim_time']}s, {row['replans']} replans, wall {row['wall_s']}s)")

    print_table(rows, by="task")
    print(f"\nresults appended to {results_path}")


if __name__ == "__main__":
    main()
