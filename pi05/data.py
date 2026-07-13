"""Training samples straight out of the RoboPref episode HDF5 files.

One sample = one camera frame of one episode:

    images   both cameras at frame k, resized with pad to 224 and scaled to [-1, 1]
    tokens   "Task: {instruction}, State: {7 bins};\nAction: " (pi05 discrete state)
    state    measured joint_pos (6) + commanded gripper (1) at the control step
             nearest the frame, quantile-normalized
    actions  the next 1.0 s of recorded commands: (50, 7) sampled every 10th
             500 Hz step, quantile-normalized and zero-padded to (50, 32)

No conversion step and no second copy of the ~40 GB datasets: episodes are read
in place. Camera frames are gzip-chunked one frame per chunk, so a random read
decodes exactly the frames it needs; four dataloader workers decode far faster
than the GPU consumes batches.

The two recorded time bases line up exactly: the recorder stamps both streams
with the same clock and captures a frame only on a step, so the control index
of frame k is ``round(t_k / 0.002)`` (asserted here, not assumed).

Debug dump (phase-b sanity of the pipeline):

    uv run python -m pi05.data --dataset .../stacking_blocks_ambiguous --dump-dir /tmp/dump
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import h5py
import numpy as np
import torch

from . import utils
from .embodiment import ARX_L5, EMBODIMENTS, EmbodimentSpec, chunk_indices


@dataclasses.dataclass(frozen=True)
class EpisodeMeta:
    path: pathlib.Path
    instruction: str
    n_steps: int
    n_frames: int


class ArxChunkDataset(torch.utils.data.Dataset):
    """(episode, camera frame) -> one pi05 training sample."""

    def __init__(
        self,
        dataset_dir: str | pathlib.Path,
        tokenizer_path: str | pathlib.Path,
        norm_stats: dict[str, utils.NormStats],
        spec: EmbodimentSpec = ARX_L5,
        split: str = "train",
        episode_filter: str | None = None,
    ):
        self.spec = spec
        self.horizon = spec.horizon
        self.sim_dt = spec.control_dt
        self.norm_stats = norm_stats
        self.tokenizer = utils.PaligemmaTokenizer(tokenizer_path, max_len=spec.token_len)

        # Index pass: open each file once in the main process and close it again
        # -- h5py handles must not be carried across a dataloader fork.
        dataset_dir = pathlib.Path(dataset_dir)
        paths = sorted(dataset_dir.glob("task_*_episode_*.hdf5"))
        if episode_filter is not None:
            paths = [p for p in paths if p.stem == episode_filter]
        if not paths:
            raise FileNotFoundError(f"no matching episodes under {dataset_dir}")

        self.episodes: list[EpisodeMeta] = []
        self.index: list[tuple[int, int]] = []  # (episode idx, camera frame k)
        for path in paths:
            with h5py.File(path, "r") as f:
                if split is not None and f.attrs["split"] != split:
                    continue
                assert abs(float(f.attrs["sim_timestep"]) - self.sim_dt) < 1e-12, path
                for camera in spec.image_obs_keys.values():
                    if f"cameras/{camera}" not in f:
                        raise ValueError(
                            f"{path.name} has no camera {camera!r} -- dataset collected "
                            f"before that view existed? Use the matching embodiment "
                            f"(e.g. arx_l5 for 2-camera recordings).")
                meta = EpisodeMeta(
                    path=path,
                    instruction=str(f.attrs["instruction"]),
                    n_steps=int(f.attrs["n_steps"]),
                    n_frames=int(f.attrs["n_camera_frames"]),
                )
            # The prompt must survive tokenization unclipped. State bins vary in
            # token count ("255" is more pieces than "5"), so check the longest
            # possible state string (every bin at 255) and require a pad slot.
            _, mask = self.tokenizer.tokenize(meta.instruction, np.ones(spec.state_dim))
            assert int(mask.sum()) < spec.token_len, (
                f"prompt for {path.name} may not fit in token_len={spec.token_len}")
            self.episodes.append(meta)
            self.index.extend((len(self.episodes) - 1, k) for k in range(meta.n_frames))

        if not self.index:
            raise ValueError(f"split {split!r} matched no episodes under {dataset_dir}")
        self._handles: dict[int, h5py.File] = {}  # per-process cache, filled after fork

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, episode_idx: int) -> h5py.File:
        f = self._handles.get(episode_idx)
        if f is None:
            f = h5py.File(self.episodes[episode_idx].path, "r")
            self._handles[episode_idx] = f
        return f

    def raw_chunk(self, f: h5py.File, j: int, n_steps: int) -> np.ndarray:
        """Un-normalized (horizon, 7) action chunk starting at control step j."""
        idx = chunk_indices(j, n_steps, self.spec.stride, self.horizon)
        # One contiguous read then a numpy stride: faster than h5py fancy indexing.
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        q_cmd = np.asarray(f["actions/q_cmd"][lo:hi], dtype=np.float32)
        grip = np.asarray(f["actions/gripper"][lo:hi], dtype=np.float32)
        rel = idx - lo
        return np.concatenate([q_cmd[rel], grip[rel, None]], axis=1)

    def __getitem__(self, i: int) -> dict:
        episode_idx, k = self.index[i]
        meta = self.episodes[episode_idx]
        f = self._file(episode_idx)

        t_k = float(f["cameras/third_person/t"][k])
        j = min(int(round(t_k / self.sim_dt)), meta.n_steps - 1)

        chunk = self.raw_chunk(f, j, meta.n_steps)
        # State gripper = the command in force at step j (== chunk row 0, since
        # there is no measured gripper channel). The eval runner feeds its last
        # executed command instead -- identical outside the one chunk row
        # (20 ms) around a toggle, against 0.25-0.45 s grip phases.
        state = np.concatenate([
            np.asarray(f["proprio/joint_pos"][j], dtype=np.float32),
            chunk[0, 6:7],
        ])
        state_norm = utils.normalize_quantile(state, self.norm_stats["state"])
        tokens, token_mask = self.tokenizer.tokenize(meta.instruction, state_norm)

        chunk_norm = utils.normalize_quantile(chunk, self.norm_stats["actions"])
        actions = utils.pad_to_dim(chunk_norm, 32).astype(np.float32)

        images = {}
        for model_key, camera in self.spec.image_obs_keys.items():
            rgb = np.asarray(f[f"cameras/{camera}/rgb"][k])
            images[model_key] = utils.image_to_model_input(rgb).transpose(2, 0, 1)

        return {"images": images, "tokens": tokens, "token_mask": token_mask, "actions": actions}


def collate(batch: list[dict]) -> dict:
    image_keys = batch[0]["images"].keys()
    images = {k: torch.from_numpy(np.stack([b["images"][k] for b in batch])) for k in image_keys}
    image_masks = {k: torch.ones(len(batch), dtype=torch.bool) for k in image_keys}
    return {
        "images": images,
        "image_masks": image_masks,
        "tokens": torch.from_numpy(np.stack([b["tokens"] for b in batch])),
        "token_mask": torch.from_numpy(np.stack([b["token_mask"] for b in batch])),
        "actions": torch.from_numpy(np.stack([b["actions"] for b in batch])),
    }


def _worker_init(_worker_id: int) -> None:
    # h5py handles must not be shared across a fork: drop any the main process
    # opened (e.g. while building validation batches) so this worker opens its
    # own. The inherited descriptors are simply never used here.
    info = torch.utils.data.get_worker_info()
    info.dataset._handles = {}


def make_dataloader(
    dataset: ArxChunkDataset,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 0,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=shuffle,
        generator=generator,
        worker_init_fn=_worker_init,
    )


# --------------------------------------------------------------------------- #
# Debug dump: verify the pipeline by eye and by assertion (phase b)            #
# --------------------------------------------------------------------------- #
def _check_time_alignment(dataset: ArxChunkDataset) -> float:
    """Max |actions_t[j] - cameras_t[k]| over every frame of every episode."""
    worst = 0.0
    for episode_idx, meta in enumerate(dataset.episodes):
        f = dataset._file(episode_idx)
        cam_t = np.asarray(f["cameras/third_person/t"])
        act_t = np.asarray(f["actions/t"])
        j = np.minimum(np.round(cam_t / dataset.sim_dt).astype(int), meta.n_steps - 1)
        worst = max(worst, float(np.abs(act_t[j] - cam_t).max()))
    return worst


def main() -> None:
    from PIL import Image

    from .inference import DEFAULT_TOKENIZER
    from .norm_stats import compute_norm_stats, load_norm_stats_file

    parser = argparse.ArgumentParser(description="Dump and check a few training samples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--norm-stats", default=None,
                        help="norm_stats JSON (default: compute from the train split)")
    parser.add_argument("--embodiment", default=ARX_L5.name, choices=sorted(EMBODIMENTS))
    parser.add_argument("-n", type=int, default=8, help="samples to dump")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    if args.norm_stats:
        norm_stats = load_norm_stats_file(args.norm_stats)
    else:
        print("computing norm stats from the train split ...")
        norm_stats = compute_norm_stats(args.dataset, split="train")

    dataset = ArxChunkDataset(args.dataset, args.tokenizer, norm_stats,
                              spec=EMBODIMENTS[args.embodiment], split=args.split)
    print(f"{len(dataset.episodes)} episodes, {len(dataset)} samples "
          f"({args.split} split, {args.embodiment})")

    dump = pathlib.Path(args.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(dataset), size=args.n, replace=False).tolist()):
        sample = dataset[i]
        episode_idx, k = dataset.index[i]
        meta = dataset.episodes[episode_idx]
        stem = f"{meta.path.stem}_frame{k}"

        for model_key in sample["images"]:
            img = ((sample["images"][model_key].transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(img).save(dump / f"{stem}_{model_key}.png")

        prompt = dataset.tokenizer._sp.decode(
            sample["tokens"][sample["token_mask"]].tolist())
        (dump / f"{stem}_prompt.txt").write_text(prompt)

        # Round-trip check: unnormalize(sample) must equal the raw recording at
        # the strided indices (tail clamped to the last step).
        f = dataset._file(episode_idx)
        t_k = float(f["cameras/third_person/t"][k])
        j = min(int(round(t_k / dataset.sim_dt)), meta.n_steps - 1)
        raw = dataset.raw_chunk(f, j, meta.n_steps)
        restored = utils.unnormalize_quantile(sample["actions"], dataset.norm_stats["actions"])[:, :7]
        err = float(np.abs(restored - raw).max())
        np.save(dump / f"{stem}_chunk_raw.npy", raw)
        assert err < 1e-4, f"normalize round-trip error {err}"
        print(f"  {stem}: t={t_k:7.3f}s j={j:5d} round-trip err {err:.2e} "
              f"grip {raw[0, 6]:.3f}->{raw[-1, 6]:.3f}  |  {prompt[:60]}...")

    worst = _check_time_alignment(dataset)
    print(f"\ntime-base alignment: max |actions_t[j] - cam_t[k]| = {worst:.2e} s "
          f"over all {len(dataset)} frames")
    assert worst < 1e-3, "camera frames do not land on control steps"
    print(f"dumped {args.n} samples to {dump}")


if __name__ == "__main__":
    main()
