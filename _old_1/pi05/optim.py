"""AdamW with float32 master weights for bf16-stored trainable parameters.

The model's forward/backward runs entirely in bfloat16 (identical numerics to
inference -- no autocast, no per-expert dtype surgery), but an Adam update of
~lr = 2.5e-5 on weights of scale ~0.05 is far below bf16's relative epsilon
(2^-8): applied directly to bf16 storage, most updates would round to nothing.
So the optimizer keeps a float32 master copy of every trainable parameter and
float32 moments, updates those, and writes the result back into the bf16
parameters after each step.

``offload=True`` moves the masters and moments to pinned CPU memory (the update
itself then runs on CPU). That saves ~7 GB of GPU memory -- the difference
between fitting and not fitting on a 12 GB card -- at ~0.5-1 s per optimizer
step, which the small-GPU recipe amortizes over gradient-accumulation
micro-steps. On an 80 GB card leave it off.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MasterAdamW:
    """Decoupled AdamW over an explicit list of (name, bf16 parameter) pairs.

    Gradients are supplied by the caller as float32 tensors on the parameters'
    device (the training loop accumulates micro-batch grads into float32
    buffers), so the optimizer never reads ``param.grad``.
    """

    def __init__(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        lr: float = 2.5e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-10,
        offload: bool = False,
        ema_decay: float | None = None,
    ):
        assert named_params, "no trainable parameters"
        self.names = [name for name, _ in named_params]
        self.params = [param for _, param in named_params]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.offload = offload
        self.ema_decay = ema_decay
        self.t = 0

        def to_state(tensor: torch.Tensor) -> torch.Tensor:
            # zeros_like/clone of a pinned tensor is NOT pinned; pin explicitly.
            return tensor.cpu().pin_memory() if offload else tensor

        self.masters = [to_state(p.detach().float()) for p in self.params]
        self.exp_avg = [to_state(torch.zeros_like(p, dtype=torch.float32)) for p in self.params]
        self.exp_avg_sq = [to_state(torch.zeros_like(p, dtype=torch.float32)) for p in self.params]
        self.ema = [to_state(m.clone()) for m in self.masters] if ema_decay else None

    @torch.no_grad()
    def step(self, grads: list[torch.Tensor], lr: float, max_grad_norm: float = 1.0) -> float:
        """One update from float32 gradients. Returns the pre-clip grad norm."""
        assert len(grads) == len(self.params)
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).item()
        if math.isfinite(max_grad_norm) and norm > max_grad_norm:
            torch._foreach_mul_(grads, max_grad_norm / (norm + 1e-6))

        if self.offload:
            grads = [g.cpu() for g in grads]

        self.t += 1
        beta1, beta2 = self.betas
        torch._foreach_mul_(self.exp_avg, beta1)
        torch._foreach_add_(self.exp_avg, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(self.exp_avg_sq, beta2)
        torch._foreach_addcmul_(self.exp_avg_sq, grads, grads, value=1.0 - beta2)

        bias1 = 1.0 - beta1**self.t
        bias2 = 1.0 - beta2**self.t
        denom = torch._foreach_sqrt(self.exp_avg_sq)
        torch._foreach_div_(denom, math.sqrt(bias2))
        torch._foreach_add_(denom, self.eps)

        if self.weight_decay:
            torch._foreach_mul_(self.masters, 1.0 - lr * self.weight_decay)
        torch._foreach_addcdiv_(self.masters, self.exp_avg, denom, value=-lr / bias1)

        if self.ema is not None:
            torch._foreach_mul_(self.ema, self.ema_decay)
            torch._foreach_add_(self.ema, self.masters, alpha=1.0 - self.ema_decay)

        for param, master in zip(self.params, self.masters):
            param.data.copy_(master, non_blocking=True)
        return norm

    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "names": self.names,
            "masters": [m.cpu() for m in self.masters],
            "exp_avg": [m.cpu() for m in self.exp_avg],
            "exp_avg_sq": [m.cpu() for m in self.exp_avg_sq],
            "ema": [m.cpu() for m in self.ema] if self.ema is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        assert state["names"] == self.names, "trainable parameter set changed"
        self.t = state["t"]

        def restore(saved: list[torch.Tensor], current: list[torch.Tensor]) -> None:
            for dst, src in zip(current, saved):
                dst.copy_(src)

        restore(state["masters"], self.masters)
        restore(state["exp_avg"], self.exp_avg)
        restore(state["exp_avg_sq"], self.exp_avg_sq)
        if self.ema is not None and state.get("ema") is not None:
            restore(state["ema"], self.ema)
        # The bf16 params must match the restored masters.
        for param, master in zip(self.params, self.masters):
            param.data.copy_(master)
