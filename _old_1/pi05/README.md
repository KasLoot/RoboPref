# Pi-0.5 — from-scratch PyTorch implementation

A clean, self-contained PyTorch reimplementation of the **π₀.₅ (pi05)**
vision-language-action (VLA) model. It loads the original
`/data/models/pi05_base` checkpoint directly and runs action inference.

The code in this folder is **independent**: it does **not** import
`transformers`, `flax`, or anything from the surrounding `openpi` package. The
only third-party pieces it relies on are `torch`, `numpy`, `sentencepiece`,
`Pillow`, and — solely to *read* the checkpoint, which is stored in Orbax's
OCDBT format — `orbax-checkpoint`/`jax` (forced onto the CPU).

---

## What the model is

π₀.₅ predicts a short *chunk* of continuous robot actions from camera images, a
language instruction, and the robot's proprioceptive state. It has three parts:

1. **SigLIP So400m/14 vision encoder** — each `224×224` image becomes `256`
   tokens, projected to the language-model width (`2048`). Up to three camera
   views are used (`base`, `left_wrist`, `right_wrist`).
2. **Gemma "mixture-of-experts" transformer** (18 layers). Two experts share a
   single attention operation but keep separate weights:
   - *expert 0* — the 2048-dim PaliGemma backbone, which processes the image and
     text tokens (the "prefix");
   - *expert 1* — a 1024-dim **action expert**, which processes the action
     tokens (the "suffix").
3. **Flow-matching action head** — starting from Gaussian noise, it integrates a
   learned velocity field for a few steps (default 10) to produce the action
   chunk.

### pi05-specific details (vs pi0)

- **State as discrete tokens.** The (normalized) robot state is discretized into
  256 bins and written into the language prompt as text, e.g.
  `Task: pick up the cube, State: 127 42 ... ;\nAction:`. There is no continuous
  "state" input token.
- **adaptive RMSNorm (adaRMS).** The flow-matching timestep is embedded by a
  small MLP and injected into every action-expert norm as per-feature
  `scale`/`shift`/`gate` modulation (the `*_norm_1/Dense_0` weights in the
  checkpoint).

---

## Files

| File            | Contents |
|-----------------|----------|
| `model.py`      | The full model: `SiglipVisionModel`, the two-expert `GemmaMixture`, the flow-matching head, `Pi05Model.sample_actions`, the training loss `Pi05Model.compute_loss`, and `Pi05Model.from_pretrained` (Orbax → PyTorch weight loading). |
| `utils.py`      | Checkpoint reading, quantile (un)normalization, image resize-with-pad, and the PaliGemma/SentencePiece tokenizer with the pi05 discrete-state prompt format. |
| `inference.py`  | A runnable demo plus a reusable `Pi05Policy` class (`infer_raw` for any embodiment, `infer` for the DROID observation dict). |
| `embodiment.py` | Per-robot conventions (`ARX_L5`): camera→image-slot mapping, state/action layout, chunk timing. |
| `norm_stats.py` | Per-dataset state/action normalization statistics (mean/std/q01/q99). |
| `data.py`       | `ArxChunkDataset`: training samples straight out of the RoboPref episode HDF5 files. |
| `optim.py`      | `MasterAdamW`: fp32 master-weight AdamW for the bf16 parameters (optional CPU offload). |
| `train.py`      | Action-expert fine-tuning loop: frozen-prefix flow-matching, validation, checkpointing. |
| `README.md`     | This file. |

---

## How to run inference

From the repository root:

```bash
# Random DROID-style example
uv run python -m pi05.inference

# With your own instruction / more integration steps
uv run python -m pi05.inference --prompt "pick up the cup" --num-steps 10
```

Expected output:

```
Loading Pi-0.5 from /data/models/pi05_base on cuda ...
Running inference for prompt: 'pick up the cup'
Actions shape: (50, 8)
First predicted action: [-0.0025 -0.0309 0.141 0.4086 0.0411 -1.1466 -0.1239 0.5019]
```

### Options

```
--checkpoint   Checkpoint directory (default: /data/models/pi05_base).
               Must contain params/ and assets/<embodiment>/norm_stats.json.
--embodiment   Which norm-stats to use (default: droid).
--tokenizer    Path to paligemma_tokenizer.model
               (default: ~/.cache/openpi/big_vision/paligemma_tokenizer.model).
--prompt       Language instruction.
--num-steps    Flow-matching integration steps (default: 10).
--device       cuda or cpu.
```

### Programmatic use

```python
from pi05.inference import Pi05Policy, make_droid_example

policy = Pi05Policy("/data/models/pi05_base_pytorch", embodiment="droid")
actions = policy.infer(make_droid_example(prompt="stack the blocks"))
print(actions.shape)  # (50, 8)  ->  (action_horizon, DROID action dim)
```

---

## Fine-tuning on the RoboPref datasets

Three independent fine-tunes, one per dataset (ambiguous / ordered /
decomposed). Only the **action expert** is trained -- 430 M of the 3.35 B
parameters (expert-1 attention/MLP/adaRMS + the action and timestep heads);
the SigLIP tower and the PaliGemma backbone stay frozen and their prefix runs
under `no_grad`. The objective is the pi05 flow-matching loss, byte-compatible
with `sample_actions` (see `Pi05Model.compute_loss`).

Samples come straight out of the episode HDF5 files (no conversion): one
sample per camera frame -- the embodiment's 640×480 views, the instruction +
discretized 7-dim state (6 measured joints + commanded gripper) as the prompt,
and the next 1.0 s of recorded commands as the target chunk (50 rows at 50 Hz,
i.e. every 10th 500 Hz step, zero-padded to the model's 32 action dims).

### Pipeline (per dataset)

Datasets from `RoboPref_dataset_v3` onward carry three cameras (low front
`third_person`, near-top-down `top_cam`, `wrist_cam`) -- train those with
`--embodiment arx_l5_3cam`. The older 2-camera recordings map to `arx_l5`.

```bash
DS=/data/datasets/RoboPref_dataset_v3/stacking_blocks_ambiguous

# 1. normalization stats (~1 min; also computed automatically by train)
uv run python -m pi05.norm_stats --dataset $DS

# 2. optional: eyeball a few samples + pipeline assertions
uv run python -m pi05.data --dataset $DS --dump-dir /tmp/dump -n 8 --embodiment arx_l5_3cam

# 3. train (big-GPU recipe; ~3-4 h on an H100 with the 3-camera prefix)
uv run python -m pi05.train --dataset $DS --embodiment arx_l5_3cam \
    --checkpoint-out /data/models/pi05_arx_ambiguous_v3

# repeat 1+3 with stacking_blocks_ordered -> pi05_arx_ordered_v3
#            and stacking_blocks_decomposed -> pi05_arx_decomposed_v3
```

The output directory is a normal pi05 checkpoint (`model.pt` +
`assets/<embodiment>/norm_stats.json`), loadable by the unchanged
`from_pretrained` / `Pi05Policy(ckpt, embodiment=...)`; the sim runner
auto-detects the embodiment from that assets directory. `best/` holds the
checkpoint with the lowest validation action-MAE; `train_log.jsonl` has the
curves; `--resume` continues an interrupted run.

### Recipes

|                    | H100/H200/B200 (default flags)   | RTX 4070 Ti 12 GB (smoke tests) |
|--------------------|----------------------------------|---------------------------------|
| batch              | `--micro-batch 32 --accum 1`     | `--micro-batch 2 --accum 16`    |
| optimizer state    | on GPU                           | `--offload-optim` (pinned CPU)  |
| measured VRAM      | ~25 GB expected                  | 8.8 GB at micro-batch 2         |
| 15 k steps         | ~2-3 h                           | ~1-2 days (don't)               |

15 k steps × effective batch 32 ≈ 2.5 epochs. Extend with `--steps 30000` if
the validation action-MAE hasn't flattened. Sanity recipe (validated locally):
`--overfit-episode task_1_episode_1 --steps 400 --warmup 50 --lr 1e-4` must
drive the loss near zero and regurgitate the demo's chunks.

### Training on a remote GPU box

Copy: this repo (`git clone` + `uv sync`), the three v3 dataset directories
(~290 GB -- `rsync` the `stacking_blocks_*` dirs; if the dataset dir is a
git-LFS repo, **exclude the `.git/` cache**, it doubles the transfer),
`/data/models/pi05_base_pytorch/` (6.7 GB), and the tokenizer
`~/.cache/openpi/big_vision/paligemma_tokenizer.model` (4 MB). Every path is a
CLI flag, so any layout works. On Blackwell (B200) confirm the installed torch
wheel supports `sm_100`; use a cu128+ build if not. Copy the three checkpoint
dirs back for evaluation.

### Closed-loop evaluation in the simulation

```bash
# oracle sanity first (no policy): recorded episodes through the exact
# execution path -- validated 10/10 across all categories and modes
uv run python simulation/collect/run_policy.py \
    --replay /data/datasets/RoboPref_dataset_v3/stacking_blocks_ordered/task_1_episode_5.hdf5

# benchmark a fine-tuned model on fresh (provably unseen) seeds
uv run python simulation/collect/run_policy.py --dataset ordered \
    --checkpoint /data/models/pi05_arx_ordered_v3 --episodes 20 --video

# watch it live: MuJoCo viewer, plus a Tk canvas tiling every camera the
# policy sees (the checkpoint's camera set is auto-detected from its assets/)
uv run python simulation/collect/run_policy.py --dataset ordered --task 2 \
    --checkpoint /data/models/pi05_arx_ordered_v3 --episodes 1 --viewer --canvas
```

Two timing modes. Headless benchmarking is **synchronous** by default: physics
pauses while the VLA computes, so success rates are hardware-independent and
reproducible. `--viewer` (or `--live` headless) switches to **asynchronous**
deployment-style timing: physics runs continuously at wall-clock speed while
inference happens in a background thread. Both modes commit to each chunk for
`--execute-horizon` rows (0.5 s) before the next takes over -- uncommitted
replanning resamples the flow noise every ~0.2 s and dithers between modes on
multimodal tasks -- and in live mode the next observation is submitted early by
the estimated latency so the successor arrives on schedule, splicing in at the
row matching its observation's age (~0.2 s / 10 rows on the local 4070 Ti).
Pass `--no-live` with `--viewer` to watch the paused-physics benchmark
behavior instead.

The runner mirrors training exactly: same renderer size and cameras, same
resize, same state convention (measured joints + last commanded gripper), same
instruction strings, and the checkpoint's own norm stats. Chunks execute
through the sim's canonical 500 Hz loop (`CriticalDamper` every step, actions
are actuator setpoints -- no IK), replanning every `--execute-horizon 25` rows
(0.5 s). Category A success accepts any completed stack (its instruction never
names an order); B and C use the tasks' own predicates. Results append to
`<checkpoint>/eval/eval.jsonl`; `--video` writes per-episode MP4s (GIF if
imageio is absent).

---

## Input / output pipeline (DROID)

1. Assemble the 8-dim state (`7 joint positions + 1 gripper`) and
   **quantile-normalize** it to `[-1, 1]` using `assets/droid/norm_stats.json`.
2. Tokenize the prompt together with the discretized state.
3. Resize each image with aspect-preserving padding to `224×224` and scale to
   `[-1, 1]`; the exterior view → `base`, wrist view → `left_wrist`, and the
   `right_wrist` view is zero-filled and masked out.
4. `sample_actions` runs flow matching to produce a `(horizon, 32)` chunk.
5. **Un-normalize** the actions and keep the first 8 dimensions (DROID).

---

## Memory & precision notes

- The model has ~3.3B parameters. In **float32** it is ~13 GB, which does **not**
  fit on a 12 GB GPU, so `from_pretrained` defaults to **bfloat16** on CUDA
  (~6.7 GB) and casts on the CPU *before* moving to the device. On CPU it uses
  float32. bf16 matches the original checkpoint's compute dtype.
- Reading the 12 GB Orbax checkpoint pulls in JAX transitively; it is pinned to
  the CPU (`JAX_PLATFORMS=cpu`) so it never competes with PyTorch for the GPU.

---

## Correctness

The implementation was validated against the original openpi JAX model on
identical inputs and identical sampling noise:

| metric | value |
|--------|-------|
| correlation (all 50×32 action values) | **0.99991** |
| mean abs diff | 0.0014 |
| mean action magnitude | 0.0586 |

The small residual is consistent with bfloat16 parameter rounding (the JAX
reference keeps float32 parameters with bf16 activations). The dominant action
values match to three decimals, e.g. `-0.9023` (PyTorch) vs `-0.9049` (JAX).

---

## Weight mapping (checkpoint → module)

The checkpoint stores every transformer layer *stacked* (leading dim 18 for the
LLM, 27 for the ViT) and both Gemma experts as parallel weight sets (base names
for expert 0, a `_1` suffix for the action expert). `load_jax_weights` in
`model.py` slices each stacked array per layer and copies it in. Weights are
kept in the checkpoint's native einsum layout and contracted with explicit
`torch.einsum` calls, so the mapping from a checkpoint array to the computation
that uses it stays one-to-one and auditable. The loader asserts that every
model parameter is filled exactly once, so any mismatch fails loudly.
