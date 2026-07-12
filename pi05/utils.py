"""Utility functions for the from-scratch Pi-0.5 PyTorch implementation.

This module is intentionally dependency-light. It only relies on:
  - numpy / torch          (math + tensors)
  - orbax-checkpoint + jax (to read the original checkpoint, which is stored in
                            Orbax OCDBT format -- there is no way around this)
  - sentencepiece          (the PaliGemma tokenizer)
  - PIL                    (image resizing)

It deliberately does NOT import `transformers` or anything from the surrounding
`openpi` package, so the `refactor/` folder is self-contained.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Checkpoint loading                                                          #
# --------------------------------------------------------------------------- #
def load_orbax_params(params_path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Restore the raw parameter PyTree from an Orbax checkpoint as numpy arrays.

    The pi05 checkpoints released by Physical Intelligence are saved with Orbax
    in OCDBT format. We load them as plain numpy arrays (never touching an
    accelerator) and return a *flat* dict whose keys are the parameter paths
    joined by "/", e.g. ``"PaliGemma/llm/embedder/input_embedding"``.
    """
    import os

    # Keep JAX (pulled in transitively by orbax) on the CPU so it does not
    # preallocate the GPU that PyTorch needs for the model itself.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax  # local import: only needed to read the checkpoint
    import orbax.checkpoint as ocp

    params_path = pathlib.Path(params_path).resolve()

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        # Newer Orbax (>=0.11) returns a ``StepMetadata`` object whose tree lives
        # in ``.item_metadata``; older versions return the tree dict directly.
        tree_meta = getattr(metadata, "item_metadata", metadata)
        item = {"params": tree_meta["params"]}
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item
                ),
            ),
        )["params"]

    return _flatten_tree(restored)


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict of arrays into ``{"a/b/c": array}``."""
    flat: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flat.update(_flatten_tree(value, child_prefix))
    else:
        flat[prefix] = np.asarray(tree)
    return flat


# --------------------------------------------------------------------------- #
# Normalization statistics (per-embodiment)                                   #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class NormStats:
    """Quantile statistics used to normalize state / actions into ~[-1, 1]."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def load_norm_stats(assets_dir: str | pathlib.Path, embodiment: str) -> dict[str, NormStats]:
    """Load ``norm_stats.json`` for a given embodiment (e.g. ``"droid"``).

    The file lives at ``<checkpoint>/assets/<embodiment>/norm_stats.json`` and
    contains statistics for the ``state`` and ``actions`` keys.
    """
    path = pathlib.Path(assets_dir) / embodiment / "norm_stats.json"
    with path.open("r") as f:
        raw = json.load(f)["norm_stats"]

    stats = {}
    for key, values in raw.items():
        stats[key] = NormStats(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
            q01=np.asarray(values["q01"], dtype=np.float32),
            q99=np.asarray(values["q99"], dtype=np.float32),
        )
    return stats


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map values into [-1, 1] using the 1st/99th percentile range.

    Only the leading ``x.shape[-1]`` dimensions of the stats are used, so a
    zero-padded state can be normalized against a shorter stats vector.
    """
    q01 = stats.q01[..., : x.shape[-1]]
    q99 = stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse of :func:`normalize_quantile`.

    Dimensions beyond the length of the stats vector (i.e. the zero-padding that
    fills the action dimension up to 32) are passed through untouched.
    """
    q01, q99 = stats.q01, stats.q99
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Zero-pad ``x`` along ``axis`` up to ``target_dim`` (no-op if already large enough)."""
    current = x.shape[axis]
    if current >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target_dim - current)
    return np.pad(x, pad_width, constant_values=value)


# --------------------------------------------------------------------------- #
# Image preprocessing                                                         #
# --------------------------------------------------------------------------- #
def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8/float image to ``(height, width, 3)``.

    The aspect ratio is preserved by resizing to fit and zero-padding the rest
    (matching openpi's ``resize_with_pad`` behavior, which uses bilinear
    resampling of the original uint8 image).
    """
    from PIL import Image

    if image.shape[0] == height and image.shape[1] == width:
        return image

    cur_h, cur_w = image.shape[:2]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(round(cur_h / ratio))
    resized_w = int(round(cur_w / ratio))

    pil = Image.fromarray(image.astype(np.uint8))
    pil = pil.resize((resized_w, resized_h), resample=Image.BILINEAR)
    resized = np.asarray(pil)

    out = np.zeros((height, width, image.shape[2]), dtype=resized.dtype)
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    out[top : top + resized_h, left : left + resized_w] = resized
    return out


def image_to_model_input(image: np.ndarray, resolution: int = 224) -> np.ndarray:
    """Resize + scale an uint8 ``(H, W, 3)`` image to float32 ``(H, W, 3)`` in [-1, 1]."""
    image = resize_with_pad(image, resolution, resolution)
    return image.astype(np.float32) / 255.0 * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# Tokenizer (PaliGemma / SentencePiece) with the pi05 prompt format           #
# --------------------------------------------------------------------------- #
class PaligemmaTokenizer:
    """Thin wrapper around the PaliGemma SentencePiece tokenizer.

    For pi0.5 the (already normalized) robot state is discretized into 256 bins
    and embedded directly into the language prompt, producing a string like::

        Task: pick up the cube, State: 127 42 ... ;\\nAction:

    """

    def __init__(self, tokenizer_path: str | pathlib.Path, max_len: int = 200):
        import sentencepiece

        self._max_len = max_len
        with open(tokenizer_path, "rb") as f:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(token_ids, token_mask)``, both of length ``max_len``."""
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")

        # Discretize the normalized state (assumed in [-1, 1]) into 256 bins.
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized.tolist()))

        full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        tokens = self._sp.encode(full_prompt, add_bos=True)

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = self._max_len - tokens_len
            mask = [True] * tokens_len + [False] * padding
            tokens = tokens + [0] * padding
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)
