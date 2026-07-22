"""Dataset integrity checks and contact sheets.

    uv run python simulation/collect/validate.py [--root PATH] [--category a|b|c|all]
                                                 [--samples N] [--no-sheets]

Checks, per task: episode count against budget; on sampled episodes, dataset shapes
and dtypes, strictly increasing timestamps, camera timestamps that coincide with a
proprio timestamp to within one physics step, ``success == True``, and an
``executed_order`` consistent with the number of gripper close events. Then the
balance invariants (Category A: 40 episodes per order; Category C T-TOP: 40 per
stack height) and three contact sheets per task.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config as C

import h5py
import numpy as np
from PIL import Image, ImageDraw

from recorder import integrity_check
from tasks import CATEGORIES

EXPECTED = {
    "/proprio/t": ((), "float64"),
    "/proprio/joint_pos": ((6,), "float32"),
    "/proprio/joint_vel": ((6,), "float32"),
    "/proprio/joint_torque": ((6,), "float32"),
    "/proprio/ee_pos": ((3,), "float32"),
    "/proprio/ee_quat_wxyz": ((4,), "float32"),
    "/actions/t": ((), "float64"),
    "/actions/q_cmd": ((6,), "float32"),
    "/actions/gripper": ((), "float32"),
}
REQUIRED_ATTRS = ("category", "task_id", "episode_id", "instruction", "seed", "success",
                  "executed_order", "blocks_present", "init_stack", "split",
                  "sim_timestep", "tcp_offset", "quat_convention")


def check_episode(path):
    """Deep check of one episode. Returns (problems, info)."""
    problems, info = [], {}
    with h5py.File(path, "r") as f:
        for key in REQUIRED_ATTRS:
            if key not in f.attrs:
                problems.append(f"missing root attr {key}")
        if not f.attrs.get("success", False):
            problems.append("success is not True")
        if f.attrs.get("quat_convention") != "wxyz":
            problems.append("quat_convention != wxyz")

        n = f["/proprio/t"].shape[0]
        for key, (tail, dtype) in EXPECTED.items():
            ds = f[key]
            if ds.shape != (n, *tail):
                problems.append(f"{key} shape {ds.shape} != {(n, *tail)}")
            if ds.dtype.name != dtype:
                problems.append(f"{key} dtype {ds.dtype.name} != {dtype}")
            if ds.compression is None:
                problems.append(f"{key} is not compressed")
            if ds.chunks is None:
                problems.append(f"{key} is not chunked")

        t = f["/proprio/t"][:]
        dt = float(f.attrs["sim_timestep"])
        if not np.all(np.diff(t) > 0):
            problems.append("proprio timestamps not strictly increasing")
        if not np.allclose(np.diff(t), dt, atol=1e-9):
            problems.append("proprio timestamps are not one physics step apart")
        if not np.array_equal(t, f["/actions/t"][:]):
            problems.append("actions/t != proprio/t")

        for cam in C.CAMERAS:
            g = f[f"/cameras/{cam}"]
            k = g["t"].shape[0]
            if g["rgb"].shape != (k, C.CAM_HEIGHT, C.CAM_WIDTH, 3):
                problems.append(f"{cam}/rgb shape {g['rgb'].shape}")
            if g["rgb"].dtype != np.uint8:
                problems.append(f"{cam}/rgb dtype {g['rgb'].dtype}")
            if g["depth"].shape != (k, C.CAM_HEIGHT, C.CAM_WIDTH):
                problems.append(f"{cam}/depth shape {g['depth'].shape}")
            if g["depth"].dtype != np.uint16:
                problems.append(f"{cam}/depth dtype {g['depth'].dtype}")
            ct = g["t"][:]
            if not np.all(np.diff(ct) > 0):
                problems.append(f"{cam} timestamps not strictly increasing")
            # every camera stamp must land on a proprio stamp, within one step
            idx = np.searchsorted(t, ct)
            idx = np.clip(idx, 1, len(t) - 1)
            near = np.minimum(np.abs(t[idx] - ct), np.abs(t[idx - 1] - ct))
            if near.size and near.max() > dt + 1e-9:
                problems.append(f"{cam} timestamp off the physics grid by "
                                f"{near.max() * 1e3:.3f} ms")
            for key in ("fovy", "resolution", "pos", "quat_wxyz"):
                if key not in g.attrs:
                    problems.append(f"{cam} missing intrinsic/extrinsic {key}")
            info[f"{cam}_frames"] = k
            if k:
                span = ct[-1] - ct[0]
                info[f"{cam}_hz"] = (k - 1) / span if span > 0 else float("nan")

        # gripper close events must match the number of placements
        grip = f["/actions/gripper"][:]
        closing = np.flatnonzero((grip[1:] <= C.GRIP_CLOSE) & (grip[:-1] > C.GRIP_CLOSE))
        order = [c for c in str(f.attrs["executed_order"]).split(",") if c]
        if len(closing) != len(order):
            problems.append(f"{len(closing)} gripper close events != "
                            f"{len(order)} placements in executed_order")
        blocks = [c for c in str(f.attrs["blocks_present"]).split(",") if c]
        if not set(order) <= set(blocks):
            problems.append(f"executed_order {order} not a subset of blocks_present {blocks}")

        info.update(
            n_steps=n, sim_s=float(t[-1] - t[0]) if n else 0.0,
            bytes=Path(path).stat().st_size, split=str(f.attrs["split"]),
            executed_order=str(f.attrs["executed_order"]),
            init_stack=str(f.attrs["init_stack"]),
            instruction=str(f.attrs["instruction"]),
            seed=int(f.attrs["seed"]), attempts=int(f.attrs.get("attempts", 1)),
        )
    return problems, info


def contact_sheet(path, out_png):
    """First / middle / last frame of both cameras, RGB over colorized depth."""
    with h5py.File(path, "r") as f:
        cams = list(C.CAMERAS)
        k = f[f"/cameras/{cams[0]}/t"].shape[0]
        picks = [0, k // 2, k - 1]
        rows = []
        for cam in cams:
            rgb = [f[f"/cameras/{cam}/rgb"][i] for i in picks]
            dep = [f[f"/cameras/{cam}/depth"][i] for i in picks]
            rows.append((f"{cam} rgb", rgb))
            rows.append((f"{cam} depth", [_colorize(d) for d in dep]))
        stamps = [float(f[f"/cameras/{cams[0]}/t"][i]) for i in picks]
        title = (f"{Path(path).name}   {f.attrs['instruction']}\n"
                 f"order={f.attrs['executed_order']}  init_stack="
                 f"{f.attrs['init_stack'] or '-'}  split={f.attrs['split']}  "
                 f"seed={f.attrs['seed']}")

    h, w, pad, top = C.CAM_HEIGHT // 2, C.CAM_WIDTH // 2, 6, 46
    sheet = Image.new("RGB", (3 * w + 4 * pad, len(rows) * (h + 18) + top + pad), "white")
    draw = ImageDraw.Draw(sheet)
    draw.multiline_text((pad, 4), title, fill="black")
    for r, (label, imgs) in enumerate(rows):
        y = top + r * (h + 18)
        draw.text((pad, y), f"{label}   t = " + ", ".join(f"{s:.2f}s" for s in stamps),
                  fill="black")
        for c, im in enumerate(imgs):
            sheet.paste(Image.fromarray(im).resize((w, h)), (pad + c * (w + pad), y + 14))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def _colorize(depth_mm):
    """uint16 mm -> grayscale RGB, invalid (0) shown as black."""
    valid = depth_mm > 0
    out = np.zeros(depth_mm.shape, dtype=np.uint8)
    if valid.any():
        d = depth_mm[valid].astype(np.float32)
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        out[valid] = np.clip(255 * (hi - d) / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


def validate_task(module, task_id, root, n_samples, sheets):
    out_dir = root / module.NAME
    budget = module.BUDGET[task_id]
    files = sorted(out_dir.glob(f"task_{task_id}_episode_*.hdf5"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    problems = []
    if len(files) != budget:
        problems.append(f"{len(files)} episodes, budget {budget}")

    quick_bad = [p.name for p in files if not integrity_check(p)[0]]
    if quick_bad:
        problems.append(f"{len(quick_bad)} failed the fast check: {quick_bad[:3]}")

    if not files:
        return {"task": task_id, "n": 0, "problems": problems, "infos": [],
                "orders": Counter(), "heights": Counter(), "bytes": 0}

    step = max(len(files) // max(n_samples, 1), 1)
    sampled = files[::step][:n_samples]
    infos = []
    for path in sampled:
        probs, info = check_episode(path)
        problems += [f"{path.name}: {p}" for p in probs]
        infos.append(info)

    # balance invariants read every file's attrs (cheap: attrs only)
    orders, heights, total_bytes = Counter(), Counter(), 0
    for path in files:
        total_bytes += path.stat().st_size
        with h5py.File(path, "r") as f:
            orders[str(f.attrs["executed_order"])] += 1
            stack = str(f.attrs["init_stack"])
            heights[len([c for c in stack.split(",") if c])] += 1

    if sheets:
        for path in ([files[0], files[len(files) // 2], files[-1]] if len(files) >= 3
                     else files):
            contact_sheet(path, out_dir / "contact_sheets" /
                          f"{path.stem}_sheet.png")

    return {"task": task_id, "n": len(files), "problems": problems, "infos": infos,
            "orders": orders, "heights": heights, "bytes": total_bytes}


def balance_problems(module, task_id, rep):
    out = []
    if module.CATEGORY == "a" and rep["n"] == module.BUDGET[task_id]:
        want = module.BUDGET[task_id] // 6
        for order, n in sorted(rep["orders"].items()):
            if n != want:
                out.append(f"order {order}: {n} episodes, expected {want}")
        if len(rep["orders"]) != 6:
            out.append(f"{len(rep['orders'])} distinct orders, expected 6")
    if module.CATEGORY == "c" and task_id in module.TOP_TASKS and rep["n"] == module.BUDGET[task_id]:
        want = module.BUDGET[task_id] // 3
        for h in (1, 2, 3):
            if rep["heights"][h] != want:
                out.append(f"stack height {h}: {rep['heights'][h]} episodes, expected {want}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=C.DATASET_ROOT)
    p.add_argument("--category", choices=["a", "b", "c", "all"], default="all")
    p.add_argument("--samples", type=int, default=3, help="episodes deep-checked per task")
    p.add_argument("--no-sheets", action="store_true")
    args = p.parse_args(argv)

    cats = list(CATEGORIES) if args.category == "all" else [args.category]
    all_problems, rows, grand_bytes = [], [], 0

    for cat in cats:
        module = CATEGORIES[cat]
        if not (args.root / module.NAME / "scene.yaml").exists():
            all_problems.append(f"{module.NAME}/scene.yaml is missing")
        for task_id in sorted(module.BUDGET):
            rep = validate_task(module, task_id, args.root, args.samples, not args.no_sheets)
            rep["problems"] += balance_problems(module, task_id, rep)
            all_problems += [f"[{cat}/task{task_id}] {p}" for p in rep["problems"]]
            grand_bytes += rep["bytes"]

            infos = rep["infos"]
            splits = Counter(i["split"] for i in infos)
            rows.append((
                f"{cat}/{task_id}", rep["n"], module.BUDGET[task_id],
                np.mean([i["sim_s"] for i in infos]) if infos else 0.0,
                np.mean([i["n_steps"] for i in infos]) if infos else 0.0,
                np.mean([i["third_person_frames"] for i in infos]) if infos else 0.0,
                np.mean([i["third_person_hz"] for i in infos]) if infos else 0.0,
                rep["bytes"] / max(rep["n"], 1) / 1e6,
                len(rep["problems"]), dict(splits),
            ))

    print("=" * 118)
    print(f"{'cat/task':<9} {'eps':>5} {'budget':>7} {'sim s':>8} {'steps':>8} "
          f"{'frames':>7} {'cam Hz':>7} {'MB/ep':>8} {'probs':>6}  sampled splits")
    print("-" * 118)
    for r in rows:
        print(f"{r[0]:<9} {r[1]:>5} {r[2]:>7} {r[3]:>8.2f} {r[4]:>8.0f} {r[5]:>7.0f} "
              f"{r[6]:>7.2f} {r[7]:>8.1f} {r[8]:>6}  {r[9]}")
    print("-" * 118)
    print(f"total episodes {sum(r[1] for r in rows)}  "
          f"total storage {grand_bytes / 1e9:.2f} GB")

    if all_problems:
        print(f"\n{len(all_problems)} PROBLEM(S):")
        for prob in all_problems:
            print("  -", prob)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
