"""Registered-reference core for Krea 2 outpainting in ComfyUI.

The reference-attention machinery here (t=0 reference modulation, isolated-ref
K/V precompute, block/attention forwards) is adapted from ostris's
ComfyUI-Krea2-Ostris-Edit (MIT License, Copyright (c) 2026 Ostris, LLC):
https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit

The ONE thing that differs from a plain edit reference is `_pack_refs`: when a
normalized bbox is supplied, the reference tokens are placed into the *target*
latent grid at that box (center-sampled fractional coordinates) instead of their
own grid starting at (0, 0). This mirrors `_pack_reference_latents` in the
yijunwang2/krea2-outpaint diffusers pipeline (Apache-2.0) exactly, so the
outpaint LoRA sees the registration geometry it was trained with. The reference
still sits on RoPE frame axis i+1 -- it is NOT coplanar with the target.

Everything else (frame axis, t=0 conditioning, kv_cache) is byte-for-byte the
generic edit path, which is why the outpaint LoRA is "just" an edit LoRA plus
this coordinate change.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.common_dit
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


# ---------------------------------------------------------------------------
# Reference packing: generic (own grid) OR registered (mapped into target grid)
# ---------------------------------------------------------------------------


def _target_grid(x, patch):
    """Padded latent-patch grid (h_, w_) for a noisy latent ``x`` (4D or 5D)."""
    hs, ws = x.shape[-2], x.shape[-1]
    hp = ((hs + patch - 1) // patch) * patch
    wp = ((ws + patch - 1) // patch) * patch
    return hp // patch, wp // patch


def _pack_refs(dit, ref_latents, bs, device, dtype, target_hw=None, bbox_norm=None):
    """Patchify reference latents into tokens + RoPE positions.

    Each reference gets RoPE axis-0 index ``i + 1`` (its own frame). Then:

    * ``bbox_norm is None`` -> axis-1/axis-2 are its own y/x grid from 0
      (ostris "index" placement, used for normal edit LoRAs);
    * ``bbox_norm = [x0, y0, x1, y1]`` (normalized 0..1) -> axis-1/axis-2 map the
      reference grid into ``target_hw`` at that box with center-sampled spacing,
      matching yijunwang2/krea2-outpaint's registered placement.

    Returns ``(reftok (B, Lr, C*p*p), refpos (B, Lr, 3))``.
    """
    patch = dit.patch
    ref_tokens = []
    ref_pos = []
    for i, ref in enumerate(ref_latents):
        if ref.ndim == 5:  # (B, C, T, H, W) Wan21 layout, T == 1 for images
            rb, rc, rt, rh5, rw5 = ref.shape
            ref = ref.reshape(rb * rt, rc, rh5, rw5)
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref.to(device, dtype), (patch, patch))
        ref = comfy.utils.repeat_to_batch_size(ref, bs)
        rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
        ref_tokens.append(
            rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        )
        rid = torch.zeros(rh, rw, 3, device=device, dtype=torch.float32)
        rid[..., 0] = i + 1.0
        if bbox_norm is None:
            rid[..., 1] = torch.arange(rh, device=device, dtype=torch.float32)[:, None]
            rid[..., 2] = torch.arange(rw, device=device, dtype=torch.float32)[None, :]
        else:
            if target_hw is None:
                raise ValueError("target_hw is required for a registered reference.")
            th, tw = target_hw
            x0, y0, x1, y1 = (float(v) for v in bbox_norm)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"Invalid normalized reference bbox: {bbox_norm}")
            ys = y0 * th + (
                torch.arange(rh, device=device, dtype=torch.float32) + 0.5
            ) * ((y1 - y0) * th / rh) - 0.5
            xs = x0 * tw + (
                torch.arange(rw, device=device, dtype=torch.float32) + 0.5
            ) * ((x1 - x0) * tw / rw) - 0.5
            rid[..., 1] = ys[:, None]
            rid[..., 2] = xs[None, :]
        ref_pos.append(rid.reshape(1, rh * rw, 3).repeat(bs, 1, 1))
    return torch.cat(ref_tokens, dim=1), torch.cat(ref_pos, dim=1)


# ---------------------------------------------------------------------------
# Attention / block forwards with K/V capture or cached-K/V injection
# ---------------------------------------------------------------------------


def _attn_kv(attn, x, freqs, mask=None, kv_capture=None, kv_cache=None,
             transformer_options={}):
    """comfy krea2 Attention.forward with optional K/V capture or cached-K/V
    injection. Both happen post-RoPE and pre-GQA-expansion."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if kv_capture is not None:
        kv_capture.append((k, v))
    if kv_cache is not None:
        k = torch.cat((k, kv_cache[0].to(k.dtype)), dim=2)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), dim=2)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = optimized_attention_masked(q, k, v, attn.heads, mask=mask,
                                     skip_reshape=True,
                                     transformer_options=transformer_options)
    return attn.wo(out * F.sigmoid(gate))


def _block_kv_forward(block, x, vec, freqs, kv_capture=None, kv_cache=None,
                      transformer_options={}):
    """SingleStreamBlock forward (single-span modulation) with K/V capture /
    injection in the attention."""
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    x = x + pregate * _attn_kv(
        block.attn, (1 + prescale) * block.prenorm(x) + preshift, freqs,
        kv_capture=kv_capture, kv_cache=kv_cache,
        transformer_options=transformer_options,
    )
    x = x + postgate * block.mlp((1 + postscale) * block.postnorm(x) + postshift)
    return x


def _block_ref_forward(block, x, vec, refvec, split, freqs, transformer_options):
    """SingleStreamBlock forward with per-span modulation: tokens ``[:split]``
    (text + noisy image) use ``vec`` (real t), tokens ``[split:]`` (clean
    reference tokens) use ``refvec`` (t=0)."""
    m = block.mod(vec)
    r = block.mod(refvec)

    def mod(h, scale, shift):
        return torch.cat(
            (
                (1 + m[scale]) * h[:, :split] + m[shift],
                (1 + r[scale]) * h[:, split:] + r[shift],
            ),
            dim=1,
        )

    def gate(h, g):
        return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

    x = x + gate(
        block.attn(
            mod(block.prenorm(x), 0, 1),
            freqs,
            None,
            transformer_options=transformer_options,
        ),
        2,
    )
    x = x + gate(block.mlp(mod(block.postnorm(x), 3, 4)), 5)
    return x


# ---------------------------------------------------------------------------
# Full forwards
# ---------------------------------------------------------------------------


def _forward_with_refs(self, x, timesteps, context, ref_latents, transformer_options,
                       bbox_norm=None):
    """Krea 2 SingleStreamDiT forward with reference latents appended at t=0
    (joint pass). With ``bbox_norm`` the refs are registered into the target
    grid. Use only for LoRAs NOT trained with isolated-ref attention; the
    outpaint LoRA wants the cached path below."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = self._unpack_context(context)

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    reftok, refpos = _pack_refs(
        self, ref_latents, bs, device, x.dtype, target_hw=(h_, w_), bbox_norm=bbox_norm
    )
    reflen = reftok.shape[1]

    img = self.first(torch.cat((img, reftok), dim=1))

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)
    t0 = self.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype)
    )
    tvec0 = self.tproj(t0)

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)
    split = txtlen + imglen - reflen

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat((txtpos, imgpos, refpos), dim=1)

    freqs = self.pe_embedder(pos)

    for block in self.blocks:
        combined = _block_ref_forward(
            block, combined, tvec, tvec0, split, freqs, transformer_options
        )

    final = self.last(combined, t)
    out = final[:, txtlen:split, :]
    out = rearrange(
        out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_, w=w_, ph=patch, pw=patch, c=self.channels,
    )
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _precompute_ref_kv(dit, x, timesteps, ref_latents, transformer_options, bbox_norm=None):
    """Run only the clean reference tokens through the blocks at t=0 and record
    each block's post-RoPE K/V. With ``bbox_norm`` the refs are registered into
    the target grid derived from ``x``."""
    temporal = x.ndim == 5
    bs = x.shape[0] * (x.shape[2] if temporal else 1)
    th, tw = _target_grid(x, dit.patch)
    reftok, refpos = _pack_refs(
        dit, ref_latents, bs, x.device, x.dtype, target_hw=(th, tw), bbox_norm=bbox_norm
    )
    h = dit.first(reftok)
    t0 = dit.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), dit.tdim).unsqueeze(1).to(h.dtype)
    )
    tvec0 = dit.tproj(t0)
    freqs = dit.pe_embedder(refpos)

    ref_kv = []
    for block in dit.blocks:
        cap = []
        h = _block_kv_forward(
            block, h, tvec0, freqs, kv_capture=cap, transformer_options=transformer_options
        )
        ref_kv.append(cap[0])
    return ref_kv


def _forward_with_cached_refs(self, x, timesteps, context, ref_kv, transformer_options):
    """Denoising pass with reference tokens replaced by their cached K/V: the
    live sequence is text + noisy image only, and every block appends the cached
    ref K/V as extra keys. Identical to ostris (the registration lives entirely
    in the precomputed K/V, so nothing here changes)."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = self._unpack_context(context)

    img = self.first(
        rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    )

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat((txtpos, imgpos), dim=1)

    freqs = self.pe_embedder(pos)

    for block, kv in zip(self.blocks, ref_kv):
        combined = _block_kv_forward(
            block, combined, tvec, freqs, kv_cache=kv, transformer_options=transformer_options
        )

    final = self.last(combined, t)
    out = final[:, txtlen : txtlen + imglen, :]
    out = rearrange(
        out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_, w=w_, ph=patch, pw=patch, c=self.channels,
    )
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


def _ref_fingerprint(ref_latents, bs, bbox_norm):
    """Content key for the ref K/V cache: batch size, per-ref shape + reduction
    sums, and the bbox (so a placement change invalidates stale K/V)."""
    key = [bs, tuple(round(v, 6) for v in bbox_norm) if bbox_norm is not None else None]
    for r in ref_latents:
        rf = r.float()
        key.append((tuple(r.shape), float(rf.sum()), float(rf.square().sum())))
    return tuple(key)
