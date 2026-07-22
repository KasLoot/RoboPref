"""Compute per-dataset normalization statistics for the ARX L5 embodiment.

pi05 quantile-normalizes the robot state and the action chunks into [-1, 1]
using per-embodiment ``norm_stats.json`` files (see ``utils.load_norm_stats``).
None of the embodiments shipped with the base checkpoint match this 7-dim
single-arm robot, so the stats are computed from the dataset itself:

    uv run python -m pi05.norm_stats \
        --dataset /data/datasets/RoboPref_dataset/stacking_blocks_ambiguous

Statistics are exact (no subsampling): every 500 Hz control step of every
*train*-split episode contributes, ~3M x 7 float32 rows per dataset, which fits
comfortably in RAM. The output JSON uses the same schema as the checkpoint's
``assets/<embodiment>/norm_stats.json`` files, and ``train.py`` copies it into
each fine-tuned checkpoint so the existing ``Pi05Policy`` load path works.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np

from . import utils
from .embodiment import ARX_L5


def episode_files(dataset_dir: str | pathlib.Path, split: str | None = None) -> list[pathlib.Path]:
    """Sorted per-episode HDF5 files, optionally filtered by the ``split`` attr."""
    dataset_dir = pathlib.Path(dataset_dir)
    paths = sorted(dataset_dir.glob("task_*_episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no task_*_episode_*.hdf5 files under {dataset_dir}")
    if split is None:
        return paths
    kept = []
    for path in paths:
        with h5py.File(path, "r") as f:
            if f.attrs["split"] == split:
                kept.append(path)
    return kept


def episode_state_actions(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """(state (T,7), actions (T,7)) for one episode, in the ARX_L5 layout."""
    gripper = np.asarray(f["actions/gripper"], dtype=np.float32)[:, None]
    state = np.concatenate([np.asarray(f["proprio/joint_pos"], dtype=np.float32), gripper], axis=1)
    actions = np.concatenate([np.asarray(f["actions/q_cmd"], dtype=np.float32), gripper], axis=1)
    return state, actions


def _stats(rows: np.ndarray) -> utils.NormStats:
    rows64 = rows.astype(np.float64)
    q01, q99 = np.quantile(rows64, [0.01, 0.99], axis=0)
    return utils.NormStats(
        mean=rows64.mean(axis=0).astype(np.float32),
        std=rows64.std(axis=0).astype(np.float32),
        q01=q01.astype(np.float32),
        q99=q99.astype(np.float32),
    )


def compute_norm_stats(dataset_dir: str | pathlib.Path, split: str = "train") -> dict[str, utils.NormStats]:
    states, actions = [], []
    for path in episode_files(dataset_dir, split):
        with h5py.File(path, "r") as f:
            s, a = episode_state_actions(f)
        states.append(s)
        actions.append(a)
    return {
        "state": _stats(np.concatenate(states, axis=0)),
        "actions": _stats(np.concatenate(actions, axis=0)),
    }


def write_norm_stats(stats: dict[str, utils.NormStats], out_path: str | pathlib.Path) -> None:
    """Write the exact JSON schema that ``utils.load_norm_stats`` reads."""
    payload = {
        "norm_stats": {
            key: {
                "mean": value.mean.tolist(),
                "std": value.std.tolist(),
                "q01": value.q01.tolist(),
                "q99": value.q99.tolist(),
            }
            for key, value in stats.items()
        }
    }
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_norm_stats_file(path: str | pathlib.Path) -> dict[str, utils.NormStats]:
    """Read a norm_stats JSON at an arbitrary path (vs. utils.load_norm_stats,
    which resolves ``<assets>/<embodiment>/norm_stats.json``)."""
    with pathlib.Path(path).open("r") as f:
        return utils.parse_norm_stats(json.load(f)["norm_stats"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ARX L5 norm stats for one dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset dir with task_*_episode_*.hdf5 files.")
    parser.add_argument("--split", default="train", help="Which split attr to include (default: train).")
    parser.add_argument("--out", default=None,
                        help=f"Output JSON (default: <dataset>/norm_stats_{ARX_L5.name}.json).")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset)
    out = pathlib.Path(args.out) if args.out else dataset_dir / f"norm_stats_{ARX_L5.name}.json"

    stats = compute_norm_stats(dataset_dir, split=args.split)
    write_norm_stats(stats, out)

    np.set_printoptions(precision=4, suppress=True)
    dims = [*(f"joint{i + 1}" for i in range(6)), "gripper"]
    for key, value in stats.items():
        print(f"\n{key} ({args.split} split):")
        print(f"  {'dim':<8} {'mean':>9} {'std':>9} {'q01':>9} {'q99':>9}")
        for i, dim in enumerate(dims):
            print(f"  {dim:<8} {value.mean[i]:>9.4f} {value.std[i]:>9.4f} "
                  f"{value.q01[i]:>9.4f} {value.q99[i]:>9.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
