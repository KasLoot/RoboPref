"""From-scratch PyTorch implementation of the Pi-0.5 (pi05) VLA model.

Pi-0.5 is a vision-language-action model built from three pieces:

  1. A SigLIP So400m/14 vision encoder that turns each 224x224 camera image
     into 256 tokens and projects them to the LLM width (2048).
  2. A Gemma "mixture of experts" transformer. Two experts share a single
     attention operation but have their own weights:
        - expert 0: the 2048-dim PaliGemma backbone (processes image + text).
        - expert 1: a 1024-dim "action expert" (processes the action tokens).
  3. A flow-matching action head. Starting from Gaussian noise, it integrates a
     learned velocity field for a handful of steps to produce a chunk of
     continuous actions.

pi0.5 differences from pi0 that are baked into this file:
  - The robot state is fed in as *discrete language tokens* (handled by the
    tokenizer in ``utils.py``), not as a continuous "state" token.
  - The action expert injects the flow-matching timestep through adaptive
    RMSNorm (adaRMS): a small MLP maps the timestep to per-layer
    scale/shift/gate modulation vectors.

The module can load the original JAX/Orbax checkpoint directly -- see
``Pi05Model.from_pretrained``. Weight tensors are kept in the same layout as the
checkpoint and contracted with explicit ``einsum`` calls, which keeps the
mapping from checkpoint array to computation obvious and auditable.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import utils


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclasses.dataclass(frozen=True)
class Pi05Config:
    # Gemma backbone (PaliGemma "gemma_2b").
    paligemma: GemmaConfig = GemmaConfig(
        width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256
    )
    # Action expert ("gemma_300m").
    action_expert: GemmaConfig = GemmaConfig(
        width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
    )
    # SigLIP So400m/14 vision encoder.
    vision_width: int = 1152
    vision_depth: int = 27
    vision_mlp_dim: int = 4304
    vision_num_heads: int = 16
    vision_patch_size: int = 14
    image_resolution: int = 224

    vocab_size: int = 257_152
    action_dim: int = 32
    action_horizon: int = 50  # pi05_base is trained with 50; pi05_droid uses 15.
    max_token_len: int = 200


# --------------------------------------------------------------------------- #
# Small building blocks                                                        #
# --------------------------------------------------------------------------- #
def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with the tanh approximation, matching flax's default ``nn.gelu``."""
    return F.gelu(x, approximate="tanh")


class RMSNorm(nn.Module):
    """Gemma RMSNorm. The learned multiplier is applied as ``(1 + scale)``.

    When ``adaptive`` is True (the pi05 action expert), there is no static scale
    parameter; instead a linear layer maps a conditioning vector (the embedded
    flow-matching timestep) to per-feature ``scale``, ``shift`` and ``gate``.
    The gate is returned to the caller and applied at the residual connection.
    """

    def __init__(self, width: int, adaptive: bool = False):
        super().__init__()
        self.width = width
        self.adaptive = adaptive
        if adaptive:
            # Produces [scale, shift, gate] concatenated along the feature axis.
            self.modulation = nn.Linear(width, 3 * width)
        else:
            self.scale = nn.Parameter(torch.zeros(width))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        normed = self._normalize(x)
        if not self.adaptive:
            return normed * (1.0 + self.scale), None

        assert cond is not None, "adaptive RMSNorm requires a conditioning vector"
        scale, shift, gate = self.modulation(cond)[:, None, :].chunk(3, dim=-1)
        normed = normed * (1.0 + scale) + shift
        return normed, gate


# --------------------------------------------------------------------------- #
# Rotary position embeddings (RoPE)                                           #
# --------------------------------------------------------------------------- #
def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, T, H, D)`` given integer ``positions`` ``(B, T)``.

    Uses the "half split" convention (first half / second half of the feature
    dim are rotated together), matching the big_vision Gemma implementation.
    """
    dim = x.shape[-1]
    freq_exponents = (2.0 / dim) * torch.arange(dim // 2, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents  # (D/2,)
    radians = positions[..., None].float() / timescale[None, None, :]  # (B, T, D/2)
    radians = radians[..., None, :]  # (B, T, 1, D/2) -> broadcast over heads
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated.to(x.dtype)


# --------------------------------------------------------------------------- #
# Gemma mixture-of-experts transformer                                         #
# --------------------------------------------------------------------------- #
class GemmaAttention(nn.Module):
    """Joint attention shared by both experts.

    Each present expert projects its own tokens to queries/keys/values with its
    own weights; the resulting q/k/v are concatenated along the token axis so a
    single attention operation mixes information across experts. Grouped-query
    attention is used (8 query heads share 1 key/value head).
    """

    def __init__(self, configs: list[GemmaConfig]):
        super().__init__()
        self.configs = configs
        self.num_heads = configs[0].num_heads
        self.num_kv_heads = configs[0].num_kv_heads
        self.head_dim = configs[0].head_dim

        # Per-expert projection weights, stored in the checkpoint's native layout.
        self.q = nn.ParameterList()      # (num_heads, width, head_dim)
        self.kv = nn.ParameterList()     # (2, num_kv_heads, width, head_dim)
        self.out = nn.ParameterList()    # (num_heads, head_dim, width)
        for cfg in configs:
            self.q.append(nn.Parameter(torch.zeros(self.num_heads, cfg.width, self.head_dim)))
            self.kv.append(nn.Parameter(torch.zeros(2, self.num_kv_heads, cfg.width, self.head_dim)))
            self.out.append(nn.Parameter(torch.zeros(self.num_heads, self.head_dim, cfg.width)))

    def forward(self, xs, positions, attn_mask, kv_cache):
        qs, ks, vs = [], [], []
        for i, x in enumerate(xs):
            if x is None:
                continue
            q = torch.einsum("btd,ndh->btnh", x, self.q[i])
            kv = torch.einsum("bsd,zkdh->zbskh", x, self.kv[i])
            qs.append(q)
            ks.append(kv[0])
            vs.append(kv[1])

        q = torch.cat(qs, dim=1)  # (B, T, num_heads, head_dim)
        k = torch.cat(ks, dim=1)  # (B, T, num_kv_heads, head_dim)
        v = torch.cat(vs, dim=1)

        q = apply_rope(q, positions)
        q = q * (self.head_dim**-0.5)
        k = apply_rope(k, positions)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        new_cache = (k, v)

        # Grouped-query attention: split query heads into groups per kv head.
        b, t = q.shape[:2]
        s = k.shape[1]
        g = self.num_heads // self.num_kv_heads
        q = q.reshape(b, t, self.num_kv_heads, g, self.head_dim)

        logits = torch.einsum("btkgh,bskh->bkgts", q.float(), k.float())
        # attn_mask: (B, T, S) True where attention is allowed.
        mask = attn_mask[:, None, None, :, :]
        big_neg = -2.3819763e38
        logits = torch.where(mask, logits, big_neg)
        probs = torch.softmax(logits, dim=-1).to(v.dtype)

        encoded = torch.einsum("bkgts,bskh->btkgh", probs, v)
        encoded = encoded.reshape(b, t, self.num_heads, self.head_dim)

        outs = []
        start = 0
        for i, x in enumerate(xs):
            if x is None:
                outs.append(None)
                continue
            end = start + x.shape[1]
            outs.append(torch.einsum("btnh,nhd->btd", encoded[:, start:end], self.out[i]))
            start = end
        return outs, new_cache


class GemmaMLP(nn.Module):
    """Gated MLP (GeGLU) for a single Gemma expert."""

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        # gating_einsum: (2, width, mlp_dim); linear: (mlp_dim, width).
        self.gating = nn.Parameter(torch.zeros(2, cfg.width, cfg.mlp_dim))
        self.linear = nn.Parameter(torch.zeros(cfg.mlp_dim, cfg.width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.einsum("btf,fh->bth", x, self.gating[0])
        up = torch.einsum("btf,fh->bth", x, self.gating[1])
        hidden = gelu_tanh(gate) * up
        return torch.einsum("bth,hf->btf", hidden, self.linear)


class GemmaBlock(nn.Module):
    """One transformer layer, holding the (up to) two experts' sub-modules."""

    def __init__(self, configs: list[GemmaConfig], adarms: list[bool]):
        super().__init__()
        self.attn = GemmaAttention(configs)
        self.pre_attn_norm = nn.ModuleList(
            [RMSNorm(cfg.width, adaptive=a) for cfg, a in zip(configs, adarms)]
        )
        self.pre_ffw_norm = nn.ModuleList(
            [RMSNorm(cfg.width, adaptive=a) for cfg, a in zip(configs, adarms)]
        )
        self.mlp = nn.ModuleList([GemmaMLP(cfg) for cfg in configs])

    def forward(self, xs, positions, attn_mask, kv_cache, adarms_cond):
        # --- attention block ---
        pre, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                pre.append(None)
                gates.append(None)
                continue
            normed, gate = self.pre_attn_norm[i](x, adarms_cond[i])
            pre.append(normed)
            gates.append(gate)

        attn_out, new_cache = self.attn(pre, positions, attn_mask, kv_cache)
        xs = [_gated_residual(x, y, g) for x, y, g in zip(xs, attn_out, gates)]

        # --- feed-forward block ---
        out, gates = [], []
        for i, x in enumerate(xs):
            if x is None:
                out.append(None)
                gates.append(None)
                continue
            normed, gate = self.pre_ffw_norm[i](x, adarms_cond[i])
            out.append(self.mlp[i](normed))
            gates.append(gate)

        xs = [_gated_residual(x, y, g) for x, y, g in zip(xs, out, gates)]
        return xs, new_cache


def _gated_residual(x, y, gate):
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate


class GemmaMixture(nn.Module):
    """The full stacked Gemma transformer with a shared token embedder."""

    def __init__(self, configs: list[GemmaConfig], vocab_size: int, adarms: list[bool]):
        super().__init__()
        self.configs = configs
        self.embed_dim = configs[0].width
        self.embedder = nn.Parameter(torch.zeros(vocab_size, self.embed_dim))
        self.layers = nn.ModuleList(
            [GemmaBlock(configs, adarms) for _ in range(configs[0].depth)]
        )
        self.final_norm = nn.ModuleList(
            [RMSNorm(cfg.width, adaptive=a) for cfg, a in zip(configs, adarms)]
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings for expert 0 and scale by sqrt(embed_dim)."""
        emb = self.embedder[tokens]
        return emb * math.sqrt(self.embed_dim)

    def forward(self, xs, positions, attn_mask, adarms_cond=None, kv_cache=None):
        """Run the transformer.

        Args:
            xs: list ``[x0, x1]``; either may be None to skip that expert.
            positions: ``(B, T)`` integer positions for the tokens in ``xs``.
            attn_mask: ``(B, T, S)`` boolean attention mask.
            adarms_cond: list ``[cond0, cond1]`` conditioning vectors (or None).
            kv_cache: list of per-layer ``(k, v)`` tuples, or None.

        Returns:
            ``(outputs, new_kv_cache)`` where ``outputs`` mirrors ``xs``.
        """
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        new_cache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None
            xs, cache = layer(xs, positions, attn_mask, layer_cache, adarms_cond)
            new_cache.append(cache)

        outputs = []
        for i, x in enumerate(xs):
            if x is None:
                outputs.append(None)
            else:
                outputs.append(self.final_norm[i](x, adarms_cond[i])[0])
        return outputs, new_cache


# --------------------------------------------------------------------------- #
# SigLIP vision encoder                                                        #
# --------------------------------------------------------------------------- #
class SiglipMLP(nn.Module):
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, width)

    def forward(self, x):
        return self.fc2(gelu_tanh(self.fc1(x)))


class SiglipAttention(nn.Module):
    """Standard (full, non-causal) multi-head self-attention for the ViT."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        # Weights kept in flax layout: kernels (width, num_heads, head_dim).
        self.q_w = nn.Parameter(torch.zeros(width, num_heads, self.head_dim))
        self.q_b = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.k_w = nn.Parameter(torch.zeros(width, num_heads, self.head_dim))
        self.k_b = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v_w = nn.Parameter(torch.zeros(width, num_heads, self.head_dim))
        self.v_b = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.o_w = nn.Parameter(torch.zeros(num_heads, self.head_dim, width))
        self.o_b = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        q = torch.einsum("bsd,dnh->bsnh", x, self.q_w) + self.q_b
        k = torch.einsum("bsd,dnh->bsnh", x, self.k_w) + self.k_b
        v = torch.einsum("bsd,dnh->bsnh", x, self.v_w) + self.v_b

        scale = self.head_dim**-0.5
        logits = torch.einsum("btnh,bsnh->bnts", q.float(), k.float()) * scale
        probs = torch.softmax(logits, dim=-1).to(v.dtype)
        attended = torch.einsum("bnts,bsnh->btnh", probs, v)
        return torch.einsum("btnh,nhd->btd", attended, self.o_w) + self.o_b


class SiglipBlock(nn.Module):
    def __init__(self, width: int, mlp_dim: int, num_heads: int):
        super().__init__()
        self.ln0 = nn.LayerNorm(width, eps=1e-6)
        self.attn = SiglipAttention(width, num_heads)
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = SiglipMLP(width, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.ln0(x))
        x = x + self.mlp(self.ln1(x))
        return x


class SiglipVisionModel(nn.Module):
    """SigLIP So400m/14 encoder producing image tokens projected to ``out_dim``."""

    def __init__(self, cfg: Pi05Config, out_dim: int):
        super().__init__()
        w = cfg.vision_width
        self.patch_size = cfg.vision_patch_size
        num_patches = (cfg.image_resolution // cfg.vision_patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, w, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, w))
        self.blocks = nn.ModuleList(
            [SiglipBlock(w, cfg.vision_mlp_dim, cfg.vision_num_heads) for _ in range(cfg.vision_depth)]
        )
        self.encoder_norm = nn.LayerNorm(w, eps=1e-6)
        self.head = nn.Linear(w, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image``: ``(B, 3, H, W)`` in [-1, 1] -> tokens ``(B, num_patches, out_dim)``."""
        x = self.patch_embed(image.to(self.patch_embed.weight.dtype))  # (B, W, h, w)
        x = x.flatten(2).transpose(1, 2)                               # (B, num_patches, W)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.encoder_norm(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Attention-mask helper                                                        #
# --------------------------------------------------------------------------- #
def make_attn_mask(input_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
    """Build a ``(B, T, S)`` boolean attention mask from ``big_vision`` semantics.

    A token may attend to any valid token whose cumulative ``ar_mask`` is less
    than or equal to its own. ``ar_mask`` marks the start of a new causal block.
    """
    ar_mask = ar_mask.expand_as(input_mask)
    cumsum = torch.cumsum(ar_mask.int(), dim=1)
    attn = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    return attn & valid


# --------------------------------------------------------------------------- #
# Full Pi-0.5 model                                                            #
# --------------------------------------------------------------------------- #
class Pi05Model(nn.Module):
    IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config

        # Vision encoder projects to the PaliGemma (expert 0) width.
        self.vision = SiglipVisionModel(config, out_dim=config.paligemma.width)
        # Two-expert transformer: expert 0 = PaliGemma, expert 1 = action expert (adaRMS).
        self.llm = GemmaMixture(
            [config.paligemma, config.action_expert],
            vocab_size=config.vocab_size,
            adarms=[False, True],
        )

        ae_width = config.action_expert.width
        self.action_in_proj = nn.Linear(config.action_dim, ae_width)
        self.action_out_proj = nn.Linear(ae_width, config.action_dim)
        self.time_mlp_in = nn.Linear(ae_width, ae_width)
        self.time_mlp_out = nn.Linear(ae_width, ae_width)

    # ---- prefix / suffix embedding -------------------------------------- #
    def embed_prefix(self, images, image_masks, tokens, token_mask):
        """Embed images + language tokens into the shared prefix sequence.

        Image keys absent from ``images`` are skipped entirely. This is exactly
        equivalent to passing a zero image with a False mask -- masked tokens
        can neither attend nor be attended to, and positions count only valid
        tokens -- but skips their vision-encoder and attention cost.
        """
        embs, masks = [], []
        for key in self.IMAGE_KEYS:
            if key not in images:
                continue
            image_tokens = self.vision(images[key])  # (B, 256, 2048)
            embs.append(image_tokens)
            masks.append(image_masks[key][:, None].expand(-1, image_tokens.shape[1]))

        text_emb = self.llm.embed_tokens(tokens)  # (B, L, 2048)
        embs.append(text_emb)
        masks.append(token_mask)

        tokens_emb = torch.cat(embs, dim=1)
        input_mask = torch.cat(masks, dim=1)
        # The whole prefix is bidirectional (no causal blocks).
        ar_mask = torch.zeros(tokens_emb.shape[1], dtype=torch.bool, device=tokens_emb.device)
        return tokens_emb, input_mask, ar_mask

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy actions + timestep into the action-expert suffix sequence.

        Returns the suffix tokens, their input mask, the auto-regressive mask,
        and the adaRMS conditioning vector derived from the timestep.
        """
        action_tokens = self.action_in_proj(noisy_actions)  # (B, H, 1024)

        time_emb = posemb_sincos(timestep, action_tokens.shape[-1], min_period=4e-3, max_period=4.0)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        adarms_cond = time_emb  # (B, 1024)

        b, horizon = action_tokens.shape[:2]
        input_mask = torch.ones(b, horizon, dtype=torch.bool, device=action_tokens.device)
        # The action chunk forms a single causal block: it attends to the prefix
        # and within itself, but the prefix cannot attend back to it.
        ar_mask = torch.zeros(horizon, dtype=torch.bool, device=action_tokens.device)
        ar_mask[0] = True
        return action_tokens, input_mask, ar_mask, adarms_cond

    # ---- shared prefix/suffix passes ------------------------------------- #
    # compute_loss trains against the exact attention patterns sample_actions
    # executes; both MUST go through these two helpers so the mask/position
    # construction cannot drift apart.
    def _prefix_kv(self, images, image_masks, tokens, token_mask):
        """One forward pass over the prefix (expert 0 only) -> (kv_cache, prefix_mask)."""
        prefix_emb, prefix_mask, prefix_ar = self.embed_prefix(images, image_masks, tokens, token_mask)
        prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
        prefix_positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        _, kv_cache = self.llm([prefix_emb, None], prefix_positions, prefix_attn)
        return kv_cache, prefix_mask

    def _suffix_velocity(self, x_t, timestep, prefix_mask, kv_cache):
        """Velocity field for noisy actions ``x_t`` at ``timestep``, attending to
        the cached prefix. The suffix attends to all valid prefix tokens plus
        itself (bidirectionally); positions continue after the valid prefix."""
        horizon = x_t.shape[1]
        suffix_emb, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)                    # (B, H, H)
        prefix_kv_mask = prefix_mask[:, None, :].expand(-1, horizon, -1)        # (B, H, prefix_len)
        full_attn = torch.cat([prefix_kv_mask, suffix_attn], dim=-1)            # (B, H, prefix+H)
        positions = prefix_mask.int().sum(dim=-1, keepdim=True) + torch.cumsum(suffix_mask.int(), dim=-1) - 1

        (_, suffix_out), _ = self.llm(
            [None, suffix_emb], positions, full_attn,
            adarms_cond=[None, adarms_cond], kv_cache=kv_cache,
        )
        return self.action_out_proj(suffix_out[:, -horizon:])

    # ---- action sampling (flow matching) -------------------------------- #
    @torch.no_grad()
    def sample_actions(self, images, image_masks, tokens, token_mask, *, num_steps: int = 10, noise=None):
        """Generate an action chunk of shape ``(B, action_horizon, action_dim)``."""
        device = tokens.device
        batch_size = tokens.shape[0]
        horizon, adim = self.config.action_horizon, self.config.action_dim

        if noise is None:
            noise = torch.randn(batch_size, horizon, adim, device=device, dtype=self.dtype)

        # 1) Fill the KV cache with a single forward pass over the prefix.
        kv_cache, prefix_mask = self._prefix_kv(images, image_masks, tokens, token_mask)

        # 2) Integrate the velocity field from t=1 (noise) down to t=0.
        dt = -1.0 / num_steps
        x_t = noise
        time = 1.0
        while time >= -dt / 2:
            timestep = torch.full((batch_size,), time, device=device, dtype=self.dtype)
            v_t = self._suffix_velocity(x_t, timestep, prefix_mask, kv_cache)
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    # ---- flow-matching training loss ------------------------------------ #
    def compute_loss(self, images, image_masks, tokens, token_mask, actions, *,
                     time=None, noise=None, freeze_prefix: bool = True):
        """Per-sample flow-matching loss for a batch of action chunks.

        ``sample_actions`` integrates ``x_t <- x_t + dt * v`` from t=1 (noise)
        to t=0 (actions), so the marginal path is ``x_t = t*eps + (1-t)*a`` and
        the velocity target is ``u = eps - a``. The suffix pass below mirrors
        the sampling loop's mask/position construction exactly, so training and
        inference see identical attention patterns.

        With ``freeze_prefix`` the vision/backbone prefix runs under no_grad and
        its KV cache enters the suffix attention as a constant: gradients reach
        only the action expert (expert 1) and the action/time heads.

        Args:
            actions: ``(B, horizon, action_dim)`` normalized, zero-padded chunk.
            time: optional ``(B,)`` flow times in (0, 1]; sampled from the
                openpi convention ``Beta(1.5, 1)*0.999 + 0.001`` when None.
            noise: optional ``(B, horizon, action_dim)`` standard normal.

        Returns:
            ``(B,)`` float32 loss, the MSE over all horizon x action_dim dims.
        """
        actions = actions.float()
        batch_size = actions.shape[0]

        if noise is None:
            noise = torch.randn_like(actions)
        else:
            noise = noise.float()
        if time is None:
            time = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)) * 0.999 + 0.001
        time = time.to(device=actions.device, dtype=torch.float32)

        t = time[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions

        prefix_ctx = torch.no_grad() if freeze_prefix else torch.enable_grad()
        with prefix_ctx:
            kv_cache, prefix_mask = self._prefix_kv(images, image_masks, tokens, token_mask)

        v_t = self._suffix_velocity(x_t.to(self.dtype), time.to(self.dtype), prefix_mask, kv_cache)
        return ((v_t.float() - u_t) ** 2).mean(dim=(-2, -1))

    @property
    def dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

    # ---- checkpoint loading --------------------------------------------- #
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        config: Pi05Config | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "Pi05Model":
        """Build the model and load its weights.

        Two checkpoint layouts are accepted, chosen automatically:

        * A native PyTorch export (a directory containing ``model.pt``, produced
          by ``pi05.convert``): loaded directly with ``torch.load`` -- fast, and
          it does not require JAX/Orbax.
        * The original JAX/Orbax checkpoint (a directory containing ``params/``):
          the ~12 GB numpy PyTree is read and copied into the module one array at
          a time.

        The model is moved to ``device``/``dtype`` *before* the checkpoint arrays
        are read, so we never hold a full float32 copy of the ~3B-parameter model
        in CPU RAM alongside the checkpoint.
        """
        import pathlib

        config = config or Pi05Config()
        # Cast to the target dtype on the CPU first, *then* move to the device, so
        # we never place a full float32 copy on a GPU that can only hold bf16.
        model = cls(config).to(dtype=dtype)
        model = model.to(device=device).eval()

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        weights_path = checkpoint_dir / "model.pt"
        if weights_path.exists():
            # Load onto CPU and copy in-place into the (already on-device) params
            # so we never hold two full copies of the model on the GPU at once.
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            del state_dict
            return model

        params_path = checkpoint_dir / "params"
        flat = utils.load_orbax_params(params_path)
        load_jax_weights(model, flat)  # copies (and frees) arrays one at a time
        return model


# --------------------------------------------------------------------------- #
# Trainable-parameter selection                                                #
# --------------------------------------------------------------------------- #
# Expert-1 weights + adaRMS modulations + action/time heads. The count is fixed
# by the architecture; asserting it catches any silent drift in parameter names.
ACTION_EXPERT_PARAM_COUNT = 430_098_464


def action_expert_parameters(model: Pi05Model) -> list[str]:
    """Names of the parameters trained during an action-expert-only fine-tune.

    Everything the suffix pass touches: per-layer expert-1 attention / MLP /
    adaRMS modulation weights, the expert-1 final norm, the action in/out
    projections and the timestep MLP. The vision tower, the token embedder and
    all expert-0 (PaliGemma) weights stay frozen.
    """
    names = []
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if name.startswith(("action_in_proj.", "action_out_proj.",
                            "time_mlp_in.", "time_mlp_out.", "llm.final_norm.1.")):
            names.append(name)
        elif name.startswith("llm.layers.") and (
            (parts[3] == "attn" and parts[5] == "1")
            or (parts[3] in ("mlp", "pre_attn_norm", "pre_ffw_norm") and parts[4] == "1")
        ):
            names.append(name)

    params = dict(model.named_parameters())
    total = sum(params[n].numel() for n in names)
    if total != ACTION_EXPERT_PARAM_COUNT:
        raise RuntimeError(
            f"action-expert selection matched {total:,} params, "
            f"expected {ACTION_EXPERT_PARAM_COUNT:,} -- parameter names drifted?")
    return names


# --------------------------------------------------------------------------- #
# Timestep embedding                                                          #
# --------------------------------------------------------------------------- #
def posemb_sincos(pos: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """Sine-cosine embedding of scalar positions ``pos`` ``(B,)`` -> ``(B, dim)``."""
    assert dim % 2 == 0, "embedding dim must be even"
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = pos[:, None].float() * (1.0 / period * 2.0 * math.pi)[None, :]
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).to(pos.dtype)


# --------------------------------------------------------------------------- #
# Weight conversion: JAX/Orbax flat dict -> this module                        #
# --------------------------------------------------------------------------- #
def load_jax_weights(model: Pi05Model, flat: dict[str, np.ndarray]) -> None:
    """Copy every parameter from the flattened checkpoint into ``model``.

    All layer-stacked arrays in the checkpoint carry a leading layer dimension
    (18 for the LLM, 27 for the ViT); we slice per layer as we go. The function
    asserts that every model parameter is filled exactly once, so any naming or
    shape mismatch surfaces immediately.
    """
    filled: set[str] = set()

    def assign(param: nn.Parameter, array: np.ndarray, name: str):
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: model {tuple(param.shape)} vs ckpt {tuple(tensor.shape)}")
        param.data.copy_(tensor)
        filled.add(name)

    p = dict(model.named_parameters())

    # ---- SigLIP vision encoder ----
    img = "PaliGemma/img"
    # Conv kernel: flax HWIO (14,14,3,1152) -> torch OIHW (1152,3,14,14).
    assign(p["vision.patch_embed.weight"], flat[f"{img}/embedding/kernel"].transpose(3, 2, 0, 1), "vision.patch_embed.weight")
    assign(p["vision.patch_embed.bias"], flat[f"{img}/embedding/bias"], "vision.patch_embed.bias")
    assign(p["vision.pos_embed"], flat[f"{img}/pos_embedding"], "vision.pos_embed")
    assign(p["vision.encoder_norm.weight"], flat[f"{img}/Transformer/encoder_norm/scale"], "vision.encoder_norm.weight")
    assign(p["vision.encoder_norm.bias"], flat[f"{img}/Transformer/encoder_norm/bias"], "vision.encoder_norm.bias")
    # head: flax Dense kernel (1152,2048) -> torch Linear (2048,1152).
    assign(p["vision.head.weight"], flat[f"{img}/head/kernel"].T, "vision.head.weight")
    assign(p["vision.head.bias"], flat[f"{img}/head/bias"], "vision.head.bias")

    enc = f"{img}/Transformer/encoderblock"
    for i in range(model.config.vision_depth):
        pre = f"vision.blocks.{i}"
        assign(p[f"{pre}.ln0.weight"], flat[f"{enc}/LayerNorm_0/scale"][i], f"{pre}.ln0.weight")
        assign(p[f"{pre}.ln0.bias"], flat[f"{enc}/LayerNorm_0/bias"][i], f"{pre}.ln0.bias")
        assign(p[f"{pre}.ln1.weight"], flat[f"{enc}/LayerNorm_1/scale"][i], f"{pre}.ln1.weight")
        assign(p[f"{pre}.ln1.bias"], flat[f"{enc}/LayerNorm_1/bias"][i], f"{pre}.ln1.bias")
        att = f"{enc}/MultiHeadDotProductAttention_0"
        assign(p[f"{pre}.attn.q_w"], flat[f"{att}/query/kernel"][i], f"{pre}.attn.q_w")
        assign(p[f"{pre}.attn.q_b"], flat[f"{att}/query/bias"][i], f"{pre}.attn.q_b")
        assign(p[f"{pre}.attn.k_w"], flat[f"{att}/key/kernel"][i], f"{pre}.attn.k_w")
        assign(p[f"{pre}.attn.k_b"], flat[f"{att}/key/bias"][i], f"{pre}.attn.k_b")
        assign(p[f"{pre}.attn.v_w"], flat[f"{att}/value/kernel"][i], f"{pre}.attn.v_w")
        assign(p[f"{pre}.attn.v_b"], flat[f"{att}/value/bias"][i], f"{pre}.attn.v_b")
        assign(p[f"{pre}.attn.o_w"], flat[f"{att}/out/kernel"][i], f"{pre}.attn.o_w")
        assign(p[f"{pre}.attn.o_b"], flat[f"{att}/out/bias"][i], f"{pre}.attn.o_b")
        assign(p[f"{pre}.mlp.fc1.weight"], flat[f"{enc}/MlpBlock_0/Dense_0/kernel"][i].T, f"{pre}.mlp.fc1.weight")
        assign(p[f"{pre}.mlp.fc1.bias"], flat[f"{enc}/MlpBlock_0/Dense_0/bias"][i], f"{pre}.mlp.fc1.bias")
        assign(p[f"{pre}.mlp.fc2.weight"], flat[f"{enc}/MlpBlock_0/Dense_1/kernel"][i].T, f"{pre}.mlp.fc2.weight")
        assign(p[f"{pre}.mlp.fc2.bias"], flat[f"{enc}/MlpBlock_0/Dense_1/bias"][i], f"{pre}.mlp.fc2.bias")

    # ---- Gemma LLM ----
    llm = "PaliGemma/llm"
    assign(p["llm.embedder"], flat[f"{llm}/embedder/input_embedding"], "llm.embedder")
    # Expert 0 final norm (static RMSNorm scale).
    assign(p["llm.final_norm.0.scale"], flat[f"{llm}/final_norm/scale"], "llm.final_norm.0.scale")
    # Expert 1 final norm (adaRMS modulation Dense: kernel (1024,3072)).
    assign(p["llm.final_norm.1.modulation.weight"], flat[f"{llm}/final_norm_1/Dense_0/kernel"].T, "llm.final_norm.1.modulation.weight")
    assign(p["llm.final_norm.1.modulation.bias"], flat[f"{llm}/final_norm_1/Dense_0/bias"], "llm.final_norm.1.modulation.bias")

    lay = f"{llm}/layers"
    for i in range(model.config.paligemma.depth):
        pre = f"llm.layers.{i}"
        # Attention weights, per expert (suffix "" for expert 0, "_1" for expert 1).
        for e, suffix in enumerate(["", "_1"]):
            assign(p[f"{pre}.attn.q.{e}"], flat[f"{lay}/attn/q_einsum{suffix}/w"][i], f"{pre}.attn.q.{e}")
            assign(p[f"{pre}.attn.kv.{e}"], flat[f"{lay}/attn/kv_einsum{suffix}/w"][i], f"{pre}.attn.kv.{e}")
            assign(p[f"{pre}.attn.out.{e}"], flat[f"{lay}/attn/attn_vec_einsum{suffix}/w"][i], f"{pre}.attn.out.{e}")
            mlp_key = f"{lay}/mlp{suffix}"
            assign(p[f"{pre}.mlp.{e}.gating"], flat[f"{mlp_key}/gating_einsum"][i], f"{pre}.mlp.{e}.gating")
            assign(p[f"{pre}.mlp.{e}.linear"], flat[f"{mlp_key}/linear"][i], f"{pre}.mlp.{e}.linear")
        # Expert 0 norms: static RMSNorm.
        assign(p[f"{pre}.pre_attn_norm.0.scale"], flat[f"{lay}/pre_attention_norm/scale"][i], f"{pre}.pre_attn_norm.0.scale")
        assign(p[f"{pre}.pre_ffw_norm.0.scale"], flat[f"{lay}/pre_ffw_norm/scale"][i], f"{pre}.pre_ffw_norm.0.scale")
        # Expert 1 norms: adaRMS modulation Dense (kernel (1024,3072)).
        assign(p[f"{pre}.pre_attn_norm.1.modulation.weight"], flat[f"{lay}/pre_attention_norm_1/Dense_0/kernel"][i].T, f"{pre}.pre_attn_norm.1.modulation.weight")
        assign(p[f"{pre}.pre_attn_norm.1.modulation.bias"], flat[f"{lay}/pre_attention_norm_1/Dense_0/bias"][i], f"{pre}.pre_attn_norm.1.modulation.bias")
        assign(p[f"{pre}.pre_ffw_norm.1.modulation.weight"], flat[f"{lay}/pre_ffw_norm_1/Dense_0/kernel"][i].T, f"{pre}.pre_ffw_norm.1.modulation.weight")
        assign(p[f"{pre}.pre_ffw_norm.1.modulation.bias"], flat[f"{lay}/pre_ffw_norm_1/Dense_0/bias"][i], f"{pre}.pre_ffw_norm.1.modulation.bias")

    # ---- Action head (root-level params) ----
    for name, key in [
        ("action_in_proj", "action_in_proj"),
        ("action_out_proj", "action_out_proj"),
        ("time_mlp_in", "time_mlp_in"),
        ("time_mlp_out", "time_mlp_out"),
    ]:
        assign(p[f"{name}.weight"], flat[f"{key}/kernel"].T, f"{name}.weight")
        assign(p[f"{name}.bias"], flat[f"{key}/bias"], f"{name}.bias")

    # Sanity check: every parameter must have been filled exactly once.
    missing = set(p.keys()) - filled
    if missing:
        raise RuntimeError(f"{len(missing)} parameters were not loaded, e.g. {sorted(missing)[:5]}")
