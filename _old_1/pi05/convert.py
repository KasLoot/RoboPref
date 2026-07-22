"""Convert the original JAX/Orbax Pi-0.5 checkpoint into a native PyTorch model.

This reads the ~12 GB Orbax ``params`` PyTree once (via ``Pi05Model.from_pretrained``),
then writes the module's ``state_dict`` to ``model.pt`` in the output directory and
copies the per-embodiment ``assets`` (normalization statistics) alongside it. The
result is a self-contained PyTorch checkpoint that loads without JAX/Orbax:

    python -m pi05.convert \
        --src /data/models/pi05_base \
        --dst /data/models/pi05_base_pytorch

Afterwards, point inference at the converted directory:

    python -m pi05.inference --checkpoint /data/models/pi05_base_pytorch
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import torch

from .model import Pi05Config, Pi05Model


def convert(src: pathlib.Path, dst: pathlib.Path, device: str, dtype: torch.dtype) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loading original checkpoint from {src} on {device} ({dtype}) ...")
    config = Pi05Config()
    model = Pi05Model.from_pretrained(str(src), config, dtype=dtype, device=device)

    # Move parameters to CPU for a portable, device-agnostic on-disk checkpoint.
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    weights_path = dst / "model.pt"
    print(f"Writing {len(state_dict)} tensors to {weights_path} ...")
    torch.save(state_dict, weights_path)

    # Copy the normalization assets so the converted dir is self-contained.
    src_assets = src / "assets"
    if src_assets.exists():
        dst_assets = dst / "assets"
        print(f"Copying assets {src_assets} -> {dst_assets} ...")
        shutil.copytree(src_assets, dst_assets, dirs_exist_ok=True)

    size_gb = weights_path.stat().st_size / 1e9
    print(f"Done. Saved PyTorch checkpoint ({size_gb:.2f} GB) to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pi-0.5 Orbax checkpoint to native PyTorch.")
    parser.add_argument("--src", default="/data/models/pi05_base", help="Original checkpoint dir (with params/ and assets/).")
    parser.add_argument("--dst", default="/data/models/pi05_base_pytorch", help="Output dir for the PyTorch checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Storage dtype (bfloat16 matches the checkpoint's compute dtype and the inference default).",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    convert(pathlib.Path(args.src), pathlib.Path(args.dst), args.device, dtype)


if __name__ == "__main__":
    main()
