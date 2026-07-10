"""Collect the RoboPref VLA dataset. Resumable, deterministic, headless.

    uv run python simulation/collect/collect.py \
        --category {a,b,c,all} [--task N] [--episodes N] [--root PATH] \
        [--preview] [--overwrite]

Existing episode files that pass a fast integrity check are skipped, so an
interrupted run resumes where it stopped; ``--overwrite`` redoes them. ``--preview``
attaches engine.py's viewer and camera canvas for debugging: it paces the loop to
wall clock and writes nothing, because the Tk canvas has no place in a collection run.

An attempt that times out or fails its ground-truth predicate is discarded -- the
partial file is deleted, never renamed into place -- and the episode retries with a
derived seed. If a task's failure rate climbs past MAX_FAILURE_RATE the task is
abandoned and reported rather than ground out.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# mujoco.viewer always creates a GLFW context. An EGL mujoco.Renderer -- which is what
# CameraCanvas builds -- then cannot be made current on the same thread; it raises
# "Failed to make the EGL context current" and the process dies. So a preview run puts
# the whole process on GLFW. This has to happen before config's MUJOCO_GL setdefault.
if "--preview" in sys.argv:
    os.environ["MUJOCO_GL"] = "glfw"

import config as C  # noqa: E402,F401  (must precede mujoco: it sets MUJOCO_GL)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

import expert as E  # noqa: E402
import scene_builder as SB  # noqa: E402
from ik import TCP_OFFSET  # noqa: E402
from recorder import EpisodeRecorder, integrity_check  # noqa: E402
from tasks import CATEGORIES  # noqa: E402


class EpisodeDiscarded(RuntimeError):
    """This attempt produced nothing worth keeping."""


class PreviewClosed(Exception):
    """The user shut a preview window. Deliberately not a RuntimeError, so the
    per-attempt handler in ``collect_episode`` lets it through."""


class Preview:
    """engine.py's viewer plus the camera canvas, paced to wall clock. Debug only.

    Costs ~0.65 ms of the 2 ms step budget, so the sim runs at wall-clock speed.
    """

    def __init__(self, model, data):
        import mujoco.viewer
        from camera import CameraCanvas

        self.model = model
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.canvas = CameraCanvas(model)
        self.expert = None  # set by run_episode, so we can draw the servo target
        self._last = time.time()

    @property
    def closed(self):
        return (not self.viewer.is_running()) or self.canvas.closed

    def wrap(self, on_step):
        from engine import draw_markers

        def hook(**kw):
            if on_step is not None:
                on_step(**kw)
            if self.closed:
                raise PreviewClosed("preview window closed")

            target = getattr(self.expert, "target_pos", None)
            if target is not None:
                draw_markers(self.viewer.user_scn, target,
                             self.expert.target_quat, kw["ee_pos"])
            self.viewer.sync()
            self.canvas.update(kw["data"])

            lag = self.model.opt.timestep - (time.time() - self._last)
            if lag > 0:
                time.sleep(lag)
            self._last = time.time()
        return hook

    def close(self):
        """Viewer first, then canvas. The other order double-frees GLFW and the
        interpreter segfaults on exit."""
        try:
            self.viewer.close()
        finally:
            self.canvas.close()


def run_episode(plan, make_recorder=None, preview=False):
    """Drive the expert through one episode.

    Returns ``(success, reason, sim_time, recorder)``. A pre-built stack is settled
    and verified *before* the recorder exists, so a stack that collapses on spawn
    costs no frames. The episode clock is zeroed once the scene is known valid, and
    the camera extrinsics are captured at exactly that instant.
    """
    scene = plan.scene
    data = SB.reset(scene)

    if plan.init_stack:
        SB.settle(scene, data, C.STACK_SETTLE_TIME)
        if not SB.stack_is_intact(scene, data):
            raise EpisodeDiscarded("pre-built stack not intact after settle")
    data.time = 0.0

    recorder = make_recorder(data) if make_recorder is not None else None
    pv = Preview(scene.model, data) if preview else None
    try:
        on_step = recorder.on_step if recorder is not None else None
        if pv is not None:
            on_step = pv.wrap(on_step)

        ex = E.Expert(scene, data, on_step=on_step, safe_z=plan.safe_z)
        if pv is not None:
            pv.expert = ex
        ex.hold(C.START_SETTLE_TIME)

        placed = []
        for color in plan.order:
            xy, z = E.stack_target(scene, data, plan.below(placed))
            ex.transfer(color, xy, z)
            placed.append(color)

        ex.hold(C.SETTLE_TIME)
        ok, reason = plan.verify(scene, data)
        return ok, reason, data.time, recorder
    finally:
        if pv is not None:
            pv.close()


def episode_attrs(module, task_id, episode_id, seed, plan):
    return {
        "category": module.CATEGORY,
        "category_name": module.NAME,
        "task_id": task_id,
        "episode_id": episode_id,
        "instruction": plan.instruction,
        "seed": seed,
        "executed_order": plan.executed_order,
        "blocks_present": ",".join(plan.blocks_present),
        "init_stack": ",".join(plan.init_stack),
        "split": C.split_of(episode_id),
        "sim_timestep": float(plan.scene.model.opt.timestep),
        "tcp_offset": np.asarray(TCP_OFFSET, dtype=np.float64),
        "quat_convention": C.QUAT_CONVENTION,
        "schema_version": C.SCHEMA_VERSION,
        "cross_xy": plan.scene.cross_xy.astype(np.float64),
        "block_size": C.BLOCK_SIZE,
        "nominal_camera_fps": C.CAM_FPS,
    }


def collect_episode(module, task_id, episode_id, path, preview=False):
    """Retry with derived seeds until the episode succeeds or attempts run out."""
    discards = []
    for attempt in range(C.MAX_ATTEMPTS_PER_EPISODE + 1):
        seed = C.episode_seed(module.CATEGORY, task_id, episode_id, attempt)
        recorder = None
        try:
            plan = module.build(task_id, episode_id, seed)
            make_recorder = None
            if not preview:
                attrs = episode_attrs(module, task_id, episode_id, seed, plan)
                make_recorder = lambda d: EpisodeRecorder(path, plan.scene, d, attrs)  # noqa: E731

            ok, reason, sim_t, recorder = run_episode(plan, make_recorder, preview)
            if ok:
                if recorder is not None:
                    recorder.keep({"attempts": attempt + 1,
                                   "discarded_attempts": len(discards)})
                return attempt + 1, discards, sim_t, True, "ok"
            discards.append(reason)
        except (E.ExpertFailure, EpisodeDiscarded, RuntimeError) as exc:
            discards.append(f"{type(exc).__name__}: {exc}")
        finally:
            if recorder is not None:
                recorder.close()  # removes the .partial unless keep() ran
    return C.MAX_ATTEMPTS_PER_EPISODE + 1, discards, 0.0, False, "attempts exhausted"


def write_scene_yaml(module, root):
    path = root / module.NAME / "scene.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(module.scene_yaml(), f, sort_keys=False, default_flow_style=False)
    return path


def sweep_partials(out_dir, task_id):
    """Delete orphaned ``.partial`` files for this task.

    ``EpisodeRecorder`` removes its own partial in a ``finally``, so one only survives
    a hard kill (SIGKILL, or a crash inside the cleanup). It can never be mistaken for
    an episode -- it does not match the ``*.hdf5`` glob -- but it costs ~130 MB.
    Scoped to one task so sharded runs do not delete each other's work in progress.
    """
    stale = list(out_dir.glob(f"task_{task_id}_episode_*.hdf5.partial"))
    for path in stale:
        path.unlink()
    return stale


def collect_task(module, task_id, root, n_episodes, overwrite, preview):
    budget = module.BUDGET[task_id]
    n = budget if n_episodes is None else min(n_episodes, budget)
    out_dir = root / module.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    if not preview:  # a preview run must not touch the dataset
        for path in sweep_partials(out_dir, task_id):
            tqdm.write(f"  removed orphaned partial from an interrupted run: {path.name}")

    kept = skipped = discarded = attempts = failed = 0
    sim_total = 0.0
    aborted = None
    t0 = time.time()

    label = f"{module.CATEGORY}/task{task_id}"
    bar = tqdm(range(1, n + 1), desc=label, unit="ep", dynamic_ncols=True)
    for episode_id in bar:
        path = out_dir / C.episode_filename(task_id, episode_id)
        if not preview and path.exists() and not overwrite:
            ok, _ = integrity_check(path)
            if ok:
                skipped += 1
                bar.set_postfix_str(f"kept={kept} skip={skipped} discard={discarded}")
                continue
            path.unlink()

        try:
            n_att, discards, sim_t, ok, reason = collect_episode(
                module, task_id, episode_id, path, preview)
        except PreviewClosed:
            bar.close()
            raise
        attempts += n_att
        discarded += len(discards)
        for r in discards:
            tqdm.write(f"  [{label} ep{episode_id}] discarded attempt: {r}")
        if ok:
            kept += 1
            sim_total += sim_t
        else:
            failed += 1
            tqdm.write(f"  [{label} ep{episode_id}] GAVE UP: {reason}")

        rate = discarded / max(attempts, 1)
        bar.set_postfix_str(f"kept={kept} skip={skipped} discard={discarded} fail={rate:.1%}")
        if attempts >= 10 and rate > C.MAX_FAILURE_RATE:
            aborted = (f"failure rate {rate:.1%} > {C.MAX_FAILURE_RATE:.0%} "
                       f"after {attempts} attempts")
            tqdm.write(f"  [{label}] ABORTING TASK: {aborted}")
            break
    bar.close()

    return {"category": module.CATEGORY, "task": task_id, "kept": kept,
            "skipped": skipped, "discarded": discarded, "attempts": attempts,
            "failed": failed, "aborted": aborted, "wall_s": time.time() - t0,
            "sim_s": sim_total, "target": n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", choices=["a", "b", "c", "all"], required=True)
    p.add_argument("--task", type=int, default=None, help="a single task id")
    p.add_argument("--episodes", type=int, default=None, help="cap episodes per task")
    p.add_argument("--root", type=Path, default=C.DATASET_ROOT)
    p.add_argument("--preview", action="store_true", help="viewer + canvas; writes nothing")
    p.add_argument("--overwrite", action="store_true", help="redo existing episodes")
    args = p.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    if args.preview:
        # A preview runs at wall-clock speed, so the full budget would take hours.
        if args.episodes is None:
            args.episodes = 1
        print(f"--preview: interactive debug run, no episodes will be written.\n"
              f"           {args.episodes} episode(s) per task at wall-clock speed; "
              f"close either window to stop.")

    cats = list(CATEGORIES) if args.category == "all" else [args.category]
    reports = []
    t0 = time.time()

    for cat in cats:
        module = CATEGORIES[cat]
        yaml_path = write_scene_yaml(module, args.root)
        print(f"\n=== category {cat}: {module.NAME}  ({yaml_path}) ===")
        task_ids = sorted(module.BUDGET) if args.task is None else [args.task]
        for task_id in task_ids:
            if task_id not in module.BUDGET:
                sys.exit(f"category {cat} has no task {task_id}")
            try:
                reports.append(collect_task(module, task_id, args.root, args.episodes,
                                            args.overwrite, args.preview))
            except PreviewClosed:
                print("\npreview window closed -- stopping")
                return

    print("\n" + "=" * 96)
    print(f"{'cat/task':<10} {'kept':>6} {'skip':>6} {'discard':>8} {'gaveup':>7} "
          f"{'attempts':>9} {'fail%':>7} {'sim s':>9} {'wall s':>9}  note")
    print("-" * 96)
    for r in reports:
        rate = r["discarded"] / max(r["attempts"], 1)
        print(f"{r['category']}/{r['task']:<8} {r['kept']:>6} {r['skipped']:>6} "
              f"{r['discarded']:>8} {r['failed']:>7} {r['attempts']:>9} {rate:>6.1%} "
              f"{r['sim_s']:>9.1f} {r['wall_s']:>9.1f}  {r['aborted'] or ''}")
    print("-" * 96)
    wall = time.time() - t0
    kept = sum(r["kept"] for r in reports)
    print(f"kept {kept} episodes in {wall:.1f} s wallclock")
    if any(r["aborted"] for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
