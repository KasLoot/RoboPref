"""Fine-tune the pi05 action expert on one RoboPref dataset.

The vision tower and the PaliGemma backbone stay frozen (their prefix pass runs
under no_grad); only the ~430M-parameter action expert -- expert-1 attention /
MLP / adaRMS weights plus the action and timestep heads -- is trained, with the
flow-matching objective of ``Pi05Model.compute_loss``. Forward/backward runs in
bf16 exactly like inference; ``MasterAdamW`` keeps fp32 masters.

One run per dataset (big-GPU recipe, ~2-3 h on an H100):

    uv run python -m pi05.train \
        --dataset /data/datasets/RoboPref_dataset/stacking_blocks_ambiguous \
        --checkpoint-out /data/models/pi05_arx_ambiguous

Defaults assume an 80 GB-class GPU (micro-batch 32, optimizer state on GPU).
On a 12 GB card (smoke tests only): ``--micro-batch 2 --accum 16 --offload-optim``.

The output directory is a normal pi05 checkpoint -- ``model.pt`` plus
``assets/arx_l5/norm_stats.json`` -- so ``Pi05Model.from_pretrained`` and
``Pi05Policy(checkpoint, embodiment="arx_l5")`` load it unchanged. ``best/``
holds the checkpoint with the lowest validation action-MAE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from . import utils
from .data import ArxChunkDataset, collate, make_dataloader
from .embodiment import ARX_L5
from .inference import DEFAULT_TOKENIZER
from .model import Pi05Model, action_expert_parameters
from .norm_stats import compute_norm_stats, load_norm_stats_file, write_norm_stats
from .optim import MasterAdamW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset dir with task_*_episode_*.hdf5")
    p.add_argument("--checkpoint-in", default="/data/models/pi05_base_pytorch")
    p.add_argument("--checkpoint-out", required=True)
    p.add_argument("--norm-stats", default=None,
                   help="norm_stats JSON (default: <dataset>/norm_stats_arx_l5.json, "
                        "computed if missing)")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="grad-accumulation micro-steps")
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload-optim", action="store_true",
                   help="fp32 optimizer state in pinned CPU memory (12 GB-GPU fallback)")
    p.add_argument("--ema", type=float, default=None, help="EMA decay (e.g. 0.999); off by default")
    p.add_argument("--resume", action="store_true", help="continue from <checkpoint-out>")
    p.add_argument("--overfit-episode", default=None, metavar="STEM",
                   help="train on a single episode (e.g. task_1_episode_1) -- sanity only")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=1_000)
    p.add_argument("--action-val-every", type=int, default=5_000)
    p.add_argument("--save-every", type=int, default=2_500)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--action-val-samples", type=int, default=64)
    return p.parse_args()


def cosine_lr(step: int, *, peak: float, warmup: int, total: int, final_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def batch_to_device(batch: dict, device: str, dtype: torch.dtype) -> dict:
    return {
        "images": {k: v.to(device, dtype, non_blocking=True) for k, v in batch["images"].items()},
        "image_masks": {k: v.to(device, non_blocking=True) for k, v in batch["image_masks"].items()},
        "tokens": batch["tokens"].to(device, non_blocking=True),
        "token_mask": batch["token_mask"].to(device, non_blocking=True),
        "actions": batch["actions"].to(device, non_blocking=True),
    }


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def make_val_batches(dataset: ArxChunkDataset, n_samples: int, micro: int, seed: int) -> list[dict]:
    """Fixed validation batches with frozen per-sample (time, noise), so the
    flow-MSE is comparable across checkpoints."""
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)
    batches = []
    for lo in range(0, n, micro):
        chunk = indices[lo : lo + micro]
        batch = collate([dataset[int(i)] for i in chunk])
        b, horizon, adim = batch["actions"].shape
        batch["time"] = torch.from_numpy(rng.beta(1.5, 1.0, size=b) * 0.999 + 0.001).float()
        batch["noise"] = torch.from_numpy(rng.standard_normal((b, horizon, adim))).float()
        batches.append(batch)
    return batches


@torch.no_grad()
def eval_flow_mse(model: Pi05Model, batches: list[dict], device: str) -> float:
    total, count = 0.0, 0
    for batch in batches:
        dev = batch_to_device(batch, device, model.dtype)
        loss = model.compute_loss(
            dev["images"], dev["image_masks"], dev["tokens"], dev["token_mask"],
            dev["actions"], time=batch["time"].to(device), noise=batch["noise"].to(device))
        total += float(loss.sum())
        count += loss.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def eval_action_mae(model: Pi05Model, batches: list[dict], norm_stats: dict, device: str,
                    action_dim: int, num_steps: int = 10) -> float:
    """Unnormalized MAE (first ``action_dim`` dims) of sampled vs recorded chunks."""
    total, count = 0.0, 0
    for batch in batches:
        dev = batch_to_device(batch, device, model.dtype)
        pred = model.sample_actions(
            dev["images"], dev["image_masks"], dev["tokens"], dev["token_mask"],
            num_steps=num_steps, noise=batch["noise"].to(device, model.dtype))
        pred = utils.unnormalize_quantile(pred.float().cpu().numpy(), norm_stats["actions"])
        truth = utils.unnormalize_quantile(batch["actions"].numpy(), norm_stats["actions"])
        total += float(np.abs(pred[..., :action_dim] - truth[..., :action_dim]).mean(axis=(1, 2)).sum())
        count += pred.shape[0]
    return total / max(count, 1)


# --------------------------------------------------------------------------- #
# Checkpointing                                                                #
# --------------------------------------------------------------------------- #
def save_model(model: Pi05Model, target: pathlib.Path, norm_stats_json: pathlib.Path,
               embodiment: str) -> None:
    """Write a directory loadable by the existing ``from_pretrained`` path."""
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / "model.pt.tmp"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, target / "model.pt")
    assets = target / "assets" / embodiment
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(norm_stats_json, assets / "norm_stats.json")


@contextmanager
def ema_weights(optimizer: MasterAdamW):
    """Temporarily load the EMA weights into the model's trainable params.

    Validation and checkpoint export both run inside this context when --ema is
    on, so the metric that selects ``best/`` judges exactly the weights that get
    exported. A no-op when EMA is disabled.
    """
    if optimizer.ema is None:
        yield
        return
    backup = [p.detach().clone() for p in optimizer.params]
    for param, ema in zip(optimizer.params, optimizer.ema):
        param.data.copy_(ema)
    try:
        yield
    finally:
        for param, saved in zip(optimizer.params, backup):
            param.data.copy_(saved)


def save_train_state(path: pathlib.Path, step: int, optimizer: MasterAdamW,
                     best_action_mae: float) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "best_action_mae": best_action_mae,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
    }, tmp)
    os.replace(tmp, path)


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=pathlib.Path(__file__).parent).stdout.strip()
    except OSError:
        return "unknown"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    spec = ARX_L5
    out = pathlib.Path(args.checkpoint_out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- norm stats -------------------------------------------------------- #
    norm_stats_json = pathlib.Path(args.norm_stats) if args.norm_stats else \
        pathlib.Path(args.dataset) / f"norm_stats_{spec.name}.json"
    if not norm_stats_json.exists():
        print(f"computing norm stats -> {norm_stats_json}")
        write_norm_stats(compute_norm_stats(args.dataset, split="train"), norm_stats_json)
    norm_stats = load_norm_stats_file(norm_stats_json)

    # -- data ---------------------------------------------------------------#
    overfit = args.overfit_episode is not None
    train_ds = ArxChunkDataset(
        args.dataset, args.tokenizer, norm_stats, spec=spec,
        split=None if overfit else "train", episode_filter=args.overfit_episode)
    val_ds = train_ds if overfit else ArxChunkDataset(
        args.dataset, args.tokenizer, norm_stats, spec=spec, split="val")
    print(f"train: {len(train_ds.episodes)} episodes / {len(train_ds):,} samples; "
          f"val: {len(val_ds):,} samples")

    val_batches = make_val_batches(val_ds, args.val_samples, args.micro_batch, seed=1234)
    action_val_batches = make_val_batches(val_ds, args.action_val_samples,
                                          min(args.micro_batch, 8), seed=5678)

    # -- model --------------------------------------------------------------#
    resume_dir = out if args.resume and (out / "model.pt").exists() else args.checkpoint_in
    print(f"loading {resume_dir} ...")
    model = Pi05Model.from_pretrained(str(resume_dir), dtype=torch.bfloat16, device=args.device)

    trainable_names = action_expert_parameters(model)
    trainable_set = set(trainable_names)
    all_params = dict(model.named_parameters())
    for name, param in all_params.items():
        param.requires_grad_(name in trainable_set)
    trainable = [(name, all_params[name]) for name in trainable_names]
    print(f"trainable: {sum(p.numel() for _, p in trainable):,} params "
          f"({len(trainable)} tensors); frozen: "
          f"{sum(p.numel() for n, p in all_params.items() if n not in trainable_set):,}")

    optimizer = MasterAdamW(trainable, lr=args.lr, offload=args.offload_optim,
                            ema_decay=args.ema)
    grad_bufs = [torch.zeros_like(p, dtype=torch.float32) for _, p in trainable]

    start_step, best_action_mae = 0, math.inf
    state_path = out / "train_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_action_mae = state["best_action_mae"]
        torch.set_rng_state(state["rng_torch"])
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        np.random.set_state(state["rng_numpy"])
        print(f"resumed at step {start_step} (best action-MAE {best_action_mae:.4f})")

    # Built after the resume load: the shuffle stream is reseeded per start_step
    # so a continued run does not replay epoch-0's batch order from the top.
    loader = make_dataloader(train_ds, args.micro_batch, num_workers=args.num_workers,
                             seed=args.seed + start_step)

    with (out / "config.json").open("w") as f:
        json.dump({**vars(args), "embodiment": spec.name, "git": git_revision(),
                   "norm_stats": str(norm_stats_json),
                   "train_samples": len(train_ds)}, f, indent=2, default=str)

    # -- loop -------------------------------------------------------------- #
    log_path = out / "train_log.jsonl"
    log_file = log_path.open("a")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    def run_validation(step: int) -> None:
        nonlocal best_action_mae
        with ema_weights(optimizer):  # judge and export the weights we would serve
            val_mse = eval_flow_mse(model, val_batches, args.device)
            record = {"step": step, "val_flow_mse": round(val_mse, 6)}
            if step % args.action_val_every == 0 or step == args.steps:
                mae = eval_action_mae(model, action_val_batches, norm_stats, args.device,
                                      spec.action_dim)
                record["val_action_mae"] = round(mae, 6)
                if mae < best_action_mae:
                    best_action_mae = mae
                    save_model(model, out / "best", norm_stats_json, spec.name)
                    record["best"] = True
        log(record)
        tqdm.write(f"[val @ {step}] " + " ".join(f"{k}={v}" for k, v in record.items() if k != "step"))

    data_iter = iter(loader)
    window = deque(maxlen=50)
    started = time.time()
    bar = tqdm(range(start_step, args.steps), initial=start_step, total=args.steps,
               desc=pathlib.Path(args.dataset).name, unit="step", dynamic_ncols=True)
    for step in bar:
        lr = cosine_lr(step, peak=args.lr, warmup=args.warmup, total=args.steps,
                       final_frac=args.final_lr_frac)
        step_loss = 0.0
        for _ in range(args.accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            dev = batch_to_device(batch, args.device, model.dtype)
            loss = model.compute_loss(dev["images"], dev["image_masks"], dev["tokens"],
                                      dev["token_mask"], dev["actions"]).mean() / args.accum
            loss.backward()
            step_loss += float(loss.detach())
            with torch.no_grad():
                for buf, (_, param) in zip(grad_bufs, trainable):
                    buf.add_(param.grad)
                    param.grad = None

        grad_norm = optimizer.step(grad_bufs, lr, max_grad_norm=args.grad_clip)
        for buf in grad_bufs:
            buf.zero_()

        window.append(step_loss)
        bar.set_postfix_str(f"loss={np.mean(window):.4f} lr={lr:.2e} gnorm={grad_norm:.2f}")
        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            log({"step": done, "loss": round(step_loss, 6),
                 "loss_mean50": round(float(np.mean(window)), 6),
                 "lr": lr, "grad_norm": round(grad_norm, 4),
                 "samples": done * args.micro_batch * args.accum,
                 "wall_s": round(time.time() - started, 1),
                 "cuda_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                 if torch.cuda.is_available() else None})
        if done % args.val_every == 0 or done == args.steps:
            run_validation(done)
        if done % args.save_every == 0 or done == args.steps:
            with ema_weights(optimizer):
                save_model(model, out, norm_stats_json, spec.name)
            save_train_state(state_path, done, optimizer, best_action_mae)
            tqdm.write(f"[ckpt @ {done}] saved {out}")

    bar.close()
    log_file.close()
    print(f"done: {out} (best val action-MAE {best_action_mae:.4f} -> {out / 'best'})")


if __name__ == "__main__":
    main()
