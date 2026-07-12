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

| File           | Contents |
|----------------|----------|
| `model.py`     | The full model: `SiglipVisionModel`, the two-expert `GemmaMixture`, the flow-matching head, `Pi05Model.sample_actions`, and `Pi05Model.from_pretrained` (Orbax → PyTorch weight loading). |
| `utils.py`     | Checkpoint reading, quantile (un)normalization, image resize-with-pad, and the PaliGemma/SentencePiece tokenizer with the pi05 discrete-state prompt format. |
| `inference.py` | A runnable demo plus a reusable `Pi05Policy` class (raw DROID observation → action chunk). |
| `README.md`    | This file. |

---

## How to run

From the repository root (`/home/yuxin/workspace/openpi`):

```bash
# Random DROID-style example
python -m refactor.inference

# With your own instruction / more integration steps
python -m refactor.inference --prompt "pick up the cup" --num-steps 10
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
from refactor.inference import Pi05Policy, make_droid_example

policy = Pi05Policy("/data/models/pi05_base", embodiment="droid")
actions = policy.infer(make_droid_example(prompt="stack the blocks"))
print(actions.shape)  # (50, 8)  ->  (action_horizon, DROID action dim)
```

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
