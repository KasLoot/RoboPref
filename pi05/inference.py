"""Run inference with the from-scratch PyTorch Pi-0.5 model.

Example:
    python -m refactor.inference                      # random DROID-style example
    python -m refactor.inference --prompt "pick up the cup"

This mirrors what ``openpi``'s DROID policy does, but is fully self-contained:
it does not import ``transformers`` or anything from the ``openpi`` package.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from . import utils
from .model import Pi05Config, Pi05Model

# The PaliGemma SentencePiece tokenizer shipped/cached by openpi. Override with
# --tokenizer if it lives elsewhere on your machine.
DEFAULT_TOKENIZER = pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"


class Pi05Policy:
    """Raw observation -> unnormalized action chunk, for any embodiment.

    Wraps the model together with the tokenizer and the per-embodiment
    normalization statistics. ``infer_raw`` is the generic entry point (images
    keyed by model slot); ``infer`` keeps the original DROID observation-dict
    interface on top of it.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        embodiment: str = "droid",
        tokenizer_path: str | pathlib.Path = DEFAULT_TOKENIZER,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype | None = None,
        token_len: int | None = None,
    ):
        self.device = device
        # bf16 on GPU (the ~3.3B-param model does not fit in float32 on a 12 GB
        # card, and bf16 matches the original checkpoint's compute dtype anyway);
        # float32 on CPU for accuracy.
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.config = Pi05Config()
        self.model = Pi05Model.from_pretrained(checkpoint_dir, self.config, dtype=dtype, device=device)
        # Padding shorter than max_token_len is exact (pad tokens are masked);
        # fine-tuned embodiments pass the token_len they were trained with.
        self.tokenizer = utils.PaligemmaTokenizer(
            tokenizer_path, max_len=token_len or self.config.max_token_len)
        self.norm_stats = utils.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", embodiment)

    def infer_raw(self, images: dict, state: np.ndarray, prompt: str, *,
                  num_steps: int = 10, noise=None) -> np.ndarray:
        """Predict an unnormalized ``(horizon, 32)`` action chunk.

        Args:
            images: model image slot -> (H, W, 3) uint8, e.g.
                ``{"base_0_rgb": ..., "left_wrist_0_rgb": ...}``. Only provided
                slots are embedded; a slot may also map to None, which feeds a
                zero image with a False mask (identical result, slower).
            state: raw low-dim state; quantile-normalized with the embodiment's
                stats, then discretized into the prompt.
            prompt: language instruction.

        The caller slices the action dims it uses; dims beyond the embodiment's
        stats pass through unnormalization untouched.
        """
        state = utils.normalize_quantile(np.asarray(state, dtype=np.float32),
                                         self.norm_stats["state"])
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)

        img_tensors, mask_tensors = {}, {}
        for key, img in images.items():
            present = img is not None
            if not present:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            model_input = utils.image_to_model_input(np.asarray(img))
            t = torch.from_numpy(model_input).permute(2, 0, 1)[None].to(self.device, self.model.dtype)
            img_tensors[key] = t
            mask_tensors[key] = torch.tensor([present], device=self.device)

        tok = torch.from_numpy(tokens)[None].to(self.device)
        tok_mask = torch.from_numpy(token_mask)[None].to(self.device)

        noise_t = None
        if noise is not None:
            noise_t = torch.from_numpy(np.asarray(noise, dtype=np.float32))[None].to(self.device, self.model.dtype)

        actions = self.model.sample_actions(
            img_tensors, mask_tensors, tok, tok_mask, num_steps=num_steps, noise=noise_t
        )
        actions = actions[0].float().cpu().numpy()  # (horizon, 32)
        return utils.unnormalize_quantile(actions, self.norm_stats["actions"])

    def infer(self, observation: dict, *, num_steps: int = 10, noise=None) -> np.ndarray:
        """Predict actions for a DROID observation (original interface).

        Expected keys (matching ``droid_policy.make_droid_example``):
            observation/exterior_image_1_left: (H, W, 3) uint8
            observation/wrist_image_left:      (H, W, 3) uint8
            observation/joint_position:        (7,) float
            observation/gripper_position:      (1,) float
            prompt:                            str
        """
        gripper = np.atleast_1d(np.asarray(observation["observation/gripper_position"], dtype=np.float32))
        state = np.concatenate([np.asarray(observation["observation/joint_position"], dtype=np.float32), gripper])
        images = {
            "base_0_rgb": np.asarray(observation["observation/exterior_image_1_left"]),
            "left_wrist_0_rgb": np.asarray(observation["observation/wrist_image_left"]),
            "right_wrist_0_rgb": None,  # zero-filled and masked out, as before
        }
        actions = self.infer_raw(images, state, observation["prompt"],
                                 num_steps=num_steps, noise=noise)
        # DROID uses the first 8 action dimensions (7 joint velocities + 1 gripper).
        return actions[:, :8]


def make_droid_example(seed: int = 0, prompt: str = "do something") -> dict:
    """Create a random DROID-style observation, matching openpi's dummy example."""
    rng = np.random.default_rng(seed)
    return {
        "observation/exterior_image_1_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7).astype(np.float32),
        "observation/gripper_position": rng.random(1).astype(np.float32),
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Pi-0.5 PyTorch inference demo.")
    parser.add_argument("--checkpoint", default="/data/models/pi05_base_pytorch", help="Checkpoint dir (contains params/ and assets/).")
    parser.add_argument("--embodiment", default="droid", help="Which assets/<embodiment>/norm_stats.json to use.")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching integration steps.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading Pi-0.5 from {args.checkpoint} on {args.device} ...")
    policy = Pi05Policy(
        args.checkpoint, embodiment=args.embodiment, tokenizer_path=args.tokenizer, device=args.device
    )

    example = make_droid_example(prompt=args.prompt)
    print(f"Running inference for prompt: {args.prompt!r}")
    actions = policy.infer(example, num_steps=args.num_steps)

    print("Actions shape:", actions.shape)
    print("First predicted action:", np.array2string(actions[0], precision=4))


if __name__ == "__main__":
    main()
