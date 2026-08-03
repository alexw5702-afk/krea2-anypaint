# Copyright 2026 Ostris, LLC. All rights reserved.
#
# Portions of the Krea2Transformer2DModel implementation are adapted from
# huggingface/diffusers (Apache License, Version 2.0), Copyright 2026 Krea AI
# and The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Krea2OstrisEdit -- a self-contained Hugging Face community pipeline for Krea 2
with reference-image (edit) conditioning and Ostris AI-Toolkit LoRA loading.

Everything lives in this one file so it can be hosted as a hub community
pipeline (a model repo containing just this ``pipeline.py``):

```python
import torch
from diffusers import DiffusionPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    "krea/Krea-2-Turbo",
    custom_pipeline="ostris/Krea2OstrisEdit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")  # or pipe.enable_model_cpu_offload() on GPUs with < ~40 GB VRAM

# Load an AI-Toolkit (or already-diffusers-format) Krea 2 LoRA, e.g. the style
# reference LoRA (generates the prompt in the style of the reference images).
pipe.load_lora_weights(
    "ostris/krea2_turbo_style_reference", weight_name="krea2_style_reference.safetensors"
)

image = pipe(
    "a white yeti with horns reading a book",
    image=Image.open("style_reference.png"),  # one reference image or a list of them
    num_inference_steps=8,                    # Turbo defaults; the base model wants 28 / 4.5
    guidance_scale=0.0,
    # kv_cache=True,  # reference K/V computed once and reused every step; only for
    #                 # LoRAs trained with AI-Toolkit's kv_cache model kwarg
).images[0]
image.save("output.png")
```

Reference images condition the model in two places, matching how the edit LoRAs
are trained with Ostris AI-Toolkit (and the ComfyUI-Krea2-Ostris-Edit nodes):

1. through the Qwen3-VL text encoder: each image is embedded in the user message
   ahead of the prompt via ``Picture N: <|vision_start|><|image_pad|><|vision_end|>``
   placeholders, so the text embeddings "see" the references;
2. as clean VAE latents appended after the noisy image tokens in the transformer
   sequence. They keep the flow time ``t=0`` (they are never noised) and sit on
   rotary-position frame axis ``i + 1`` -- the Kontext-style "index" placement.

Without ``image`` the pipeline is a plain Krea 2 text-to-image sampler.
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import PeftAdapterMixin
from diffusers.models import AutoencoderKLQwenImage
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor

try:
    from transformers import AutoTokenizer, Qwen3VLModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Krea2OstrisEdit requires a transformers version that ships Qwen3-VL "
        "(`transformers>=4.57`). Please upgrade transformers."
    ) from e


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# torch>=2.5 supports grouped-query attention natively in SDPA; older versions
# need the key/value heads repeated to the query head count.
_SDPA_HAS_GQA = tuple(int(re.sub(r"\D.*", "", v) or 0) for v in torch.__version__.split(".")[:2]) >= (2, 5)


# ---------------------------------------------------------------------------
# Transformer (Krea 2 single-stream MMDiT)
#
# Module tree and state-dict keys match the `Krea2Transformer2DModel` checkpoint
# layout in the `transformer/` folder of the Krea 2 hub repos, so the sharded
# weights load directly. The forward pass additionally supports clean reference
# tokens appended after the image tokens (`ref_seq_len`), which are modulated at
# flow time t=0 while the text + noisy image tokens keep the real timestep.
# ---------------------------------------------------------------------------


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: the effective multiplier is ``1 + weight``,
    matching the Krea 2 checkpoint format. Normalization runs in float32."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(
            hidden_states.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps
        )
        return hidden_states.to(dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved (even, odd) channel pairs. ``x`` is (B, H, S, D); ``cos``/``sin``
    are (S, D) in the repeat-interleaved layout produced by ``Krea2RotaryPosEmbed``."""
    x_f = x.float()
    x_rot = torch.stack((-x_f[..., 1::2], x_f[..., 0::2]), dim=-1).flatten(-2)
    return (x_f * cos + x_rot * sin).to(x.dtype)


class Krea2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: float, axes_dim: List[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ids: (seq_len, 3) rotary coordinates. Frequencies are computed in float64
        # (float32 on backends without float64 support, e.g. MPS).
        dtype = torch.float32 if ids.device.type == "mps" else torch.float64
        angles = []
        for i, dim in enumerate(self.axes_dim):
            pos = ids[:, i].to(dtype)
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=dtype, device=ids.device) / dim))
            angles.append(pos[:, None] * freqs[None, :])
        angles = torch.cat(angles, dim=-1)
        cos = angles.cos().repeat_interleave(2, dim=-1).float()
        sin = angles.sin().repeat_interleave(2, dim=-1).float()
        return cos, sin


class Krea2Attention(nn.Module):
    """Self-attention with grouped-query projections, q/k RMSNorm, rotary embeddings
    and a sigmoid output gate."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_capture: Optional[list] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2)
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query = _apply_rotary_emb(query, cos, sin)
            key = _apply_rotary_emb(key, cos, sin)

        if kv_capture is not None:
            kv_capture.append((key, value))
        if kv_cache is not None:
            # Cached reference K/V, already rotary-embedded at their original positions.
            key = torch.cat([key, kv_cache[0].to(key.dtype)], dim=2)
            value = torch.cat([value, kv_cache[1].to(value.dtype)], dim=2)

        is_gqa = self.num_heads != self.num_kv_heads
        if is_gqa and not _SDPA_HAS_GQA:
            key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        sdpa_kwargs = {"enable_gqa": True} if (is_gqa and _SDPA_HAS_GQA) else {}
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, **sdpa_kwargs)

        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm transformer block (no rotary embeddings, no time modulation) used by
    the text fusion stage."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuses the stack of tapped text-encoder hidden states into one text sequence:
    ``layerwise_blocks`` attend across the layer axis per token, a linear ``projector``
    collapses that axis, and ``refiner_blocks`` attend across the token sequence."""

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, int]],
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_capture: Optional[list] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # ``temb`` is the (B, 1, 6 * hidden_size) modulation input, or a tuple
        # ``(temb, ref_temb, split)`` for reference-image conditioning: tokens
        # ``[:split]`` (text + noisy image) are modulated with the real timestep
        # while tokens ``[split:]`` (clean reference tokens) use the t=0 embedding.
        if isinstance(temb, tuple):
            temb, ref_temb, split = temb
            m = (temb.unflatten(-1, (6, -1)) + self.scale_shift_table).unbind(-2)
            r = (ref_temb.unflatten(-1, (6, -1)) + self.scale_shift_table).unbind(-2)

            def modulate(h, scale_idx, shift_idx):
                return torch.cat(
                    (
                        (1.0 + m[scale_idx]) * h[:, :split] + m[shift_idx],
                        (1.0 + r[scale_idx]) * h[:, split:] + r[shift_idx],
                    ),
                    dim=1,
                )

            def gate(h, gate_idx):
                return torch.cat((m[gate_idx] * h[:, :split], r[gate_idx] * h[:, split:]), dim=1)

            attn_out = self.attn(
                modulate(self.norm1(hidden_states), 0, 1),
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                kv_capture=kv_capture,
                kv_cache=kv_cache,
            )
            hidden_states = hidden_states + gate(attn_out, 2)
            ff_out = self.ff(modulate(self.norm2(hidden_states), 3, 4))
            hidden_states = hidden_states + gate(ff_out, 5)
            return hidden_states

        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            kv_capture=kv_capture,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) followed by a
    two-layer MLP. Keeps the sequence dimension at size 1 so per-block modulations
    broadcast over tokens."""

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    """Projects the fused text features into the transformer width."""

    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection."""

    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    The Krea 2 single-stream MMDiT flow-matching backbone, extended with support for
    clean reference-image tokens ("edit" conditioning).

    Text conditioning enters as a stack of hidden states tapped from several layers of
    the Qwen3-VL text encoder. A small text-fusion transformer collapses the layer axis
    and refines the token sequence; the result is concatenated with the patchified
    image latents (and, optionally, packed reference latents) into a single
    ``[text, image, refs]`` sequence processed by the transformer blocks.

    When ``ref_seq_len > 0``, the last ``ref_seq_len`` tokens of ``hidden_states`` are
    clean reference tokens: they are modulated with the t=0 timestep embedding
    (Kontext-style "index_timestep_zero") and excluded from the returned velocity.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Krea2TransformerBlock", "Krea2TextFusionBlock", "Krea2FinalLayer"]
    _keep_in_fp32_modules = ["norm", "norm1", "norm2", "norm_q", "norm_k"]
    _skip_layerwise_casting_patterns = ["time_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: Tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}"
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=in_channels, eps=norm_eps)

    def precompute_ref_kv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Run only the packed clean reference tokens through the transformer blocks at t=0
        and return each block's rotary-embedded key/value pair.

        Only valid for adapters trained with AI-Toolkit's ``kv_cache`` model kwarg,
        where reference tokens attend solely to each other: their per-block K/V are
        then independent of the timestep and of the rest of the sequence, so this
        single pass serves every denoising step. Pass the result to
        ``forward(..., ref_kv_cache=...)`` with the reference tokens dropped from
        ``hidden_states`` and ``position_ids``.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, ref_seq_len, in_channels)`):
                Packed reference latents (see the pipeline's `_pack_reference_latents`).
            position_ids (`torch.Tensor` of shape `(ref_seq_len, 3)`):
                The reference tokens' rotary coordinates.
            attention_kwargs (`dict`, *optional*):
                When it contains a `scale` entry, sets the LoRA scale, matching `forward`.

        Returns:
            A list with one `(key, value)` tuple per transformer block, each of shape
            `(batch_size, num_key_value_heads, ref_seq_len, attention_head_dim)`.
        """
        lora_scale = 1.0
        if attention_kwargs is not None:
            lora_scale = attention_kwargs.get("scale", 1.0)
        if USE_PEFT_BACKEND and lora_scale != 1.0:
            scale_lora_layers(self, lora_scale)

        # Clean reference tokens are always conditioned at flow time t=0.
        timestep = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        hidden_states = self.img_in(hidden_states)
        image_rotary_emb = self.rotary_emb(position_ids)

        ref_kv = []
        for block in self.transformer_blocks:
            captured = []
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, kv_capture=captured)
            ref_kv.append(captured[0])

        if USE_PEFT_BACKEND and lora_scale != 1.0:
            unscale_lora_layers(self, lora_scale)
        return ref_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        ref_seq_len: int = 0,
        ref_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        r"""
        Predict the flow-matching velocity for the (noisy) image tokens.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_seq_len + ref_seq_len, in_channels)`):
                Packed (patchified) noisy image latents, with any packed clean reference
                latents appended at the end.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, num_text_layers, text_hidden_dim)`):
                Stack of tapped text-encoder hidden states per token.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Flow-matching time in `[0, 1]` (1 is pure noise, 0 is clean data).
            position_ids (`torch.Tensor` of shape `(text_seq_len + image_seq_len + ref_seq_len, 3)`):
                `(t, h, w)` rotary coordinates for the combined sequence. Text rows are
                all-zero; image rows hold the latent-grid coordinates; the i-th
                reference image sits on frame axis `i + 1` with its own grid.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Boolean mask marking valid text tokens.
            ref_seq_len (`int`, defaults to 0):
                Number of trailing reference tokens in `hidden_states`. They receive the
                t=0 modulation and are excluded from the output.
            ref_kv_cache (`list[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
                Per-block reference K/V from [`~Krea2Transformer2DModel.precompute_ref_kv`].
                When given, `hidden_states` / `position_ids` must not contain the
                reference tokens (`ref_seq_len == 0`); each block's attention appends the
                cached K/V as extra keys instead. Only valid for adapters trained with
                AI-Toolkit's ``kv_cache`` model kwarg (isolated reference attention).
            attention_kwargs (`dict`, *optional*):
                When it contains a `scale` entry, sets the LoRA scale applied to this
                transformer's adapters for the duration of the forward pass.

        Returns:
            The velocity tensor of shape `(batch_size, image_seq_len, in_channels)`.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}.")
        if ref_kv_cache is not None and ref_seq_len > 0:
            raise ValueError(
                "`ref_kv_cache` replaces the reference tokens; do not also append them to "
                "`hidden_states` (`ref_seq_len` must be 0)."
            )

        lora_scale = 1.0
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        if USE_PEFT_BACKEND and lora_scale != 1.0:
            scale_lora_layers(self, lora_scale)

        batch_size, image_seq_len, _ = hidden_states.shape  # includes ref tokens
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        block_temb = temb_mod
        if ref_seq_len > 0:
            # Clean reference tokens are conditioned at t=0; everything else keeps t.
            temb_zero = self.time_embed(torch.zeros_like(timestep), dtype=hidden_states.dtype)
            ref_temb_mod = self.time_mod_proj(F.gelu(temb_zero, approximate="tanh"))
            block_temb = (temb_mod, ref_temb_mod, text_seq_len + image_seq_len - ref_seq_len)

        # An all-True mask (no padded text tokens, e.g. any batch-of-1 call) is
        # equivalent to no mask; passing None keeps SDPA on its fast, low-memory
        # (flash) path instead of a mask-materializing fallback.
        if encoder_attention_mask is not None and bool(encoder_attention_mask.all()):
            encoder_attention_mask = None

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            # Key-padding masks of shape (B, 1, 1, L): padded text tokens are excluded
            # as attention keys everywhere; their own (garbage) lanes are never read
            # back and are dropped at the output slice.
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]
            if ref_kv_cache is not None:
                # Cached reference K/V are appended as extra (always-valid) keys.
                ref_mask = attention_mask.new_ones((batch_size, 1, 1, ref_kv_cache[0][0].shape[2]))
                attention_mask = torch.cat([attention_mask, ref_mask], dim=-1)

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_func = getattr(self, "_gradient_checkpointing_func", None)
                if ckpt_func is None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        block, hidden_states, block_temb, image_rotary_emb, attention_mask, use_reentrant=False
                    )
                else:
                    hidden_states = ckpt_func(block, hidden_states, block_temb, image_rotary_emb, attention_mask)
            else:
                hidden_states = block(
                    hidden_states,
                    block_temb,
                    image_rotary_emb,
                    attention_mask,
                    kv_cache=ref_kv_cache[i] if ref_kv_cache is not None else None,
                )

        hidden_states = hidden_states[:, text_seq_len : text_seq_len + image_seq_len - ref_seq_len]
        output = self.final_layer(hidden_states, temb)

        if USE_PEFT_BACKEND and lora_scale != 1.0:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


# The Krea 2 hub repos' `model_index.json` points the `transformer` component at
# `["diffusers", "Krea2Transformer2DModel"]`. Registering the vendored class into the
# diffusers namespace lets `DiffusionPipeline.from_pretrained` resolve it on diffusers
# releases that don't ship Krea 2 yet, and guarantees the loaded transformer supports
# the reference-image forward pass this pipeline needs (the class is a numerically
# identical superset of the upstream one for text-to-image).
diffusers.Krea2Transformer2DModel = Krea2Transformer2DModel


# ---------------------------------------------------------------------------
# LoRA key conversion (Ostris AI-Toolkit / reference-trainer -> diffusers/PEFT)
# ---------------------------------------------------------------------------


def _convert_non_diffusers_krea2_lora_to_diffusers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map original `krea-ai/krea-2` module names onto `Krea2Transformer2DModel`.
    Handles the `diffusion_model.` prefix (AI-Toolkit saves / ComfyUI) and the
    `base_model.model.` prefix, as well as bare module names."""
    state_dict = {
        (k[len("base_model.model.") :] if k.startswith("base_model.model.") else k): v for k, v in state_dict.items()
    }
    state_dict = {
        (k[len("diffusion_model.") :] if k.startswith("diffusion_model.") else k): v for k, v in state_dict.items()
    }

    attn_map = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
    ff_map = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}
    # The original model stores these standalone modules under abbreviated
    # `nn.Sequential`-style names.
    standalone_map = {
        "first": "img_in",
        "last.linear": "final_layer.linear",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "txtfusion.projector": "text_fusion.projector",
    }

    def convert_module(module):
        m = re.match(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)$", module)
        if m:
            idx, kind, sub = m.groups()
            if kind == "attn" and sub in attn_map:
                return f"transformer_blocks.{idx}.attn.{attn_map[sub]}"
            if kind == "mlp" and sub in ff_map:
                return f"transformer_blocks.{idx}.{ff_map[sub]}"
            return None
        m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.(\w+)$", module)
        if m:
            block, idx, kind, sub = m.groups()
            if kind == "attn" and sub in attn_map:
                return f"text_fusion.{block}.{idx}.attn.{attn_map[sub]}"
            if kind == "mlp" and sub in ff_map:
                return f"text_fusion.{block}.{idx}.{ff_map[sub]}"
            return None
        return standalone_map.get(module)

    converted_state_dict = {}
    for key in list(state_dict):
        match = re.search(r"\.(?:lora_[AB])\.weight$", key)
        if match is None:
            continue
        diffusers_module = convert_module(key[: match.start()])
        if diffusers_module is None:
            continue
        converted_state_dict[f"transformer.{diffusers_module}{key[match.start() :]}"] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError(f"Could not convert LoRA keys: {sorted(state_dict.keys())}")

    return converted_state_dict


def _normalize_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize a Krea 2 LoRA state dict to PEFT `lora_A`/`lora_B` naming and fold any
    `.alpha` tensors into `lora_B` so the effective scale is preserved."""
    state_dict = {
        k.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight"): v
        for k, v in state_dict.items()
    }
    # PEFT assumes lora_alpha == rank (scale 1.0) when no alpha is given; fold any
    # explicit alpha into lora_B instead of plumbing network_alphas through.
    for alpha_key in [k for k in state_dict if k.endswith(".alpha")]:
        base = alpha_key[: -len(".alpha")]
        a_key, b_key = base + ".lora_A.weight", base + ".lora_B.weight"
        alpha = float(state_dict.pop(alpha_key))
        if a_key in state_dict and b_key in state_dict:
            rank = state_dict[a_key].shape[0]
            if alpha != rank:
                state_dict[b_key] = state_dict[b_key] * (alpha / rank)
    return state_dict


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Krea2PipelineOutput(BaseOutput):
    """Output class for the Krea 2 pipeline.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`):
            List of `num_batches * num_images_per_prompt` denoised PIL images or a
            numpy array of shape `(batch_size, height, width, num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 6400,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class Krea2OstrisEditPipeline(DiffusionPipeline):
    r"""
    Krea 2 text-to-image / reference-image-edit pipeline with Ostris AI-Toolkit LoRA
    loading. See the module docstring for usage.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Euler flow-matching scheduler configured with the Krea 2 resolution-aware
            exponential time shift.
        vae ([`AutoencoderKLQwenImage`]):
            The Qwen-Image VAE (f8, 16 latent channels).
        text_encoder ([`~transformers.Qwen3VLModel`]):
            Qwen3-VL, including its vision tower (used to embed reference images into
            the prompt conditioning).
        tokenizer ([`~transformers.AutoTokenizer`]):
            The tokenizer paired with the text encoder.
        transformer ([`Krea2Transformer2DModel`]):
            The Krea 2 single-stream MMDiT.
        text_encoder_select_layers (`tuple[int, ...]`, *optional*):
            Indices into the text encoder's `hidden_states` tuple whose states are
            stacked per token as the transformer's text conditioning.
        is_distilled (`bool`, *optional*, defaults to `False`):
            Whether the transformer is the few-step distilled (Turbo) checkpoint. When
            `True`, a fixed timestep shift `mu=1.15` is used and the call defaults
            change to `num_inference_steps=8, guidance_scale=0.0`.
        patch_size (`int`, *optional*, defaults to 2):
            Side length of the square patches the latents are packed into.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # Default hub repo used to lazily build the Qwen3-VL processor that turns
    # reference images into vision tokens (the Krea 2 repos ship only a tokenizer).
    vl_processor_id = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen3VLModel,
        tokenizer: AutoTokenizer,
        transformer: Krea2Transformer2DModel,
        text_encoder_select_layers: Optional[Union[Tuple[int, ...], List[int]]] = None,
        is_distilled: bool = False,
        patch_size: int = 2,
    ):
        super().__init__()

        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        if text_encoder_select_layers is None:
            text_encoder_select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
        self.register_to_config(text_encoder_select_layers=tuple(text_encoder_select_layers))
        self.text_encoder_select_layers = tuple(text_encoder_select_layers)
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.register_to_config(patch_size=patch_size)
        self.patch_size = patch_size
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        # Fixed instruction template wrapped around every prompt. The system prefix is
        # fed through the encoder as context but its hidden states are sliced off.
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34

        self._vl_processor = None

    # ------------------------------------------------------------------
    # Prompt encoding (Qwen3-VL; reference images embedded via vision tokens)
    # ------------------------------------------------------------------
    @property
    def vl_processor(self):
        """Qwen3-VL AutoProcessor, loaded lazily (only needed when reference images are
        encoded into the prompt)."""
        if self._vl_processor is None:
            from transformers import AutoProcessor

            self._vl_processor = AutoProcessor.from_pretrained(self.vl_processor_id)
        return self._vl_processor

    @staticmethod
    def _to_chw_tensor(image) -> torch.Tensor:
        """Convert a PIL image / numpy array / CHW tensor to a float CHW tensor in [0, 1]."""
        if isinstance(image, torch.Tensor):
            t = image.squeeze(0) if image.ndim == 4 else image
            t = t.float()
            if t.min() < 0:  # assume [-1, 1]
                t = (t + 1.0) / 2.0
            return t.clamp(0, 1)
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        image = image.convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _prep_vl_images(self, images: List[torch.Tensor], max_pixels: int) -> List[torch.Tensor]:
        """Resize reference images for the Qwen3-VL pass: aspect-preserving downscale
        (never upscaled) to fit ``max_pixels`` total area. The MLLM only needs a coarse
        view of the references; high-res detail flows through the VAE ref latents."""
        prepped = []
        for img in images:
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt(max_pixels / (h * w)))
            nh, nw = max(round(h * scale), 28), max(round(w * scale), 28)
            if (nh, nw) != (h, w):
                img = (
                    F.interpolate(img.unsqueeze(0).float(), size=(nh, nw), mode="bicubic", antialias=True)
                    .squeeze(0)
                    .clamp(0, 1)
                )
            prepped.append(img.float())
        return prepped

    def _encode_single_prompt(
        self,
        prompt: str,
        images: Optional[List[torch.Tensor]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode one prompt (optionally with reference images embedded as vision
        tokens) into stacked Qwen3-VL hidden states of shape `(seq_len, num_text_layers,
        text_hidden_dim)` at natural (unpadded) length."""
        device = device or self._execution_device
        prefix_idx = self.prompt_template_encode_start_idx

        # The suffix is tokenized separately so it lands after the prompt tokens.
        suffix_inputs = self.tokenizer([self.prompt_template_encode_suffix], return_tensors="pt").to(device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()

        extra_inputs = {}
        if images:
            # Reference images ride in the user message ahead of the prompt via named
            # vision placeholders; the processor expands each <|image_pad|> to the
            # image's token grid.
            image_prompt = "".join(
                f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images))
            )
            text = self.prompt_template_encode_prefix + image_prompt + prompt
            # No truncation here: the expanded image-pad runs must stay intact.
            inputs = self.vl_processor(text=[text], images=list(images), return_tensors="pt", do_rescale=False).to(
                device
            )
            for k, v in inputs.items():
                if k in ("input_ids", "attention_mask"):
                    continue
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    v = v.to(self.text_encoder.dtype)
                extra_inputs[k] = v
        else:
            text = self.prompt_template_encode_prefix + prompt
            inputs = self.tokenizer(
                [text], truncation=True, max_length=max_sequence_length + prefix_idx, return_tensors="pt"
            ).to(device)

        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        # mm_token_type_ids (used for M-RoPE) must cover the appended suffix tokens
        # too; they are plain text -> type 0.
        if "mm_token_type_ids" in extra_inputs:
            tt = extra_inputs["mm_token_type_ids"]
            extra_inputs["mm_token_type_ids"] = torch.cat(
                [tt, torch.zeros_like(suffix_ids, dtype=tt.dtype)], dim=1
            )

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **extra_inputs,
        )

        hidden_states = torch.stack([outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2)
        # Drop the system-prefix tokens; what remains is (image +) prompt + suffix.
        return hidden_states[0, prefix_idx:]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: Optional[List[torch.Tensor]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts (all sharing the same reference images, if any) and right-pad
        them into a batch. Returns `(prompt_embeds, prompt_embeds_mask)` of shapes
        `(B, L, num_text_layers, D)` and `(B, L)` (bool)."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        features = [self._encode_single_prompt(p, images, max_sequence_length, device) for p in prompt]
        max_len = max(f.shape[0] for f in features)
        embeds = features[0].new_zeros(len(features), max_len, *features[0].shape[1:])
        mask = torch.zeros(len(features), max_len, dtype=torch.bool, device=device)
        for i, f in enumerate(features):
            embeds[i, : f.shape[0]] = f
            mask[i, : f.shape[0]] = True

        embeds = embeds.repeat_interleave(num_images_per_prompt, dim=0)
        mask = mask.repeat_interleave(num_images_per_prompt, dim=0)
        return embeds, mask

    # ------------------------------------------------------------------
    # Latent packing helpers
    # ------------------------------------------------------------------
    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) latents -> (B, H/p * W/p, C * p * p) tokens."""
        b, c, h, w = latents.shape
        p = self.patch_size
        latents = latents.view(b, c, h // p, p, w // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(b, (h // p) * (w // p), c * p * p)

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """(B, L, C * p * p) tokens -> (B, C, 1, H, W) latents (frame dim for the VAE)."""
        batch_size, _, channels = latents.shape
        p = self.patch_size
        h = p * (int(height) // (self.vae_scale_factor * p))
        w = p * (int(width) // (self.vae_scale_factor * p))
        latents = latents.view(batch_size, h // p, w // p, channels // (p * p), p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (p * p), 1, h, w)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return self._pack_latents(latents)

    def _encode_reference_latents(
        self,
        images: List[torch.Tensor],
        max_pixels: int,
        generator: Optional[torch.Generator],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Encode `[0, 1]` CHW reference images to normalized VAE latents, one `(C, h, w)`
        tensor per image. Each image is downscaled (aspect-preserving, never upscaled) to
        fit within `max_pixels`, then snapped so the latent grid is patchifiable."""
        snap = self.vae_scale_factor * self.patch_size
        vae_dtype = self.vae.dtype

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)

        ref_latents = []
        for img in images:
            img = img.unsqueeze(0).to(device, dtype=vae_dtype)
            h, w = img.shape[2], img.shape[3]
            if h * w > max_pixels:
                ratio = h / w
                new_h, new_w = math.sqrt(max_pixels * ratio), math.sqrt(max_pixels / ratio)
            else:
                new_h, new_w = float(h), float(w)
            new_h = max(snap, int(round(new_h / snap)) * snap)
            new_w = max(snap, int(round(new_w / snap)) * snap)
            if (new_h, new_w) != (h, w):
                img = F.interpolate(img.float(), size=(new_h, new_w), mode="bilinear").to(vae_dtype)

            img = (img * 2.0 - 1.0).unsqueeze(2)  # [0,1] -> [-1,1], add frame dim
            latent = self.vae.encode(img).latent_dist.sample(generator)
            latent = (latent - latents_mean.to(latent.device, latent.dtype)) / latents_std.to(
                latent.device, latent.dtype
            )
            ref_latents.append(latent[:, :, 0][0])  # drop frame + batch dims -> (C, h, w)
        return ref_latents

    def _pack_reference_latents(
        self,
        ref_latents: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        placements: Optional[List[Dict[str, Any]]] = None,
        target_grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patchify reference latents into `(1, ref_seq_len, C * p * p)` tokens and build
        their `(ref_seq_len, 3)` rotary coordinates. The i-th reference sits on frame
        axis `i + 1` with its own y/x grid starting at 0 (Kontext "index" placement)."""
        p = self.patch_size
        tokens, position_ids = [], []
        for i, ref in enumerate(ref_latents):
            ref = ref.unsqueeze(0).to(device, dtype)
            tokens.append(self._pack_latents(ref))
            _, _, h, w = ref.shape
            ids = torch.zeros(h // p, w // p, 3, device=device)
            ids[..., 0] = i + 1
            placement = placements[i] if placements is not None else None
            if placement is None:
                ids[..., 1] = torch.arange(h // p, device=device)[:, None]
                ids[..., 2] = torch.arange(w // p, device=device)[None, :]
            else:
                if target_grid_size is None:
                    raise ValueError("`target_grid_size` is required for registered references.")
                bbox = placement.get("bbox_normalized")
                if bbox is None or len(bbox) != 4:
                    raise ValueError("A registered reference requires bbox_normalized=[x0,y0,x1,y1].")
                x0, y0, x1, y1 = (float(value) for value in bbox)
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError(f"Invalid normalized reference bbox: {bbox}")
                target_h, target_w = target_grid_size
                ref_h, ref_w = h // p, w // p
                ys = y0 * target_h + (torch.arange(ref_h, device=device) + 0.5) * (
                    (y1 - y0) * target_h / ref_h
                ) - 0.5
                xs = x0 * target_w + (torch.arange(ref_w, device=device) + 0.5) * (
                    (x1 - x0) * target_w / ref_w
                ) - 0.5
                ids[..., 1] = ys[:, None]
                ids[..., 2] = xs[None, :]
            position_ids.append(ids.reshape(-1, 3))
        return torch.cat(tokens, dim=1), torch.cat(position_ids, dim=0)

    @staticmethod
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        """Rotary coordinates for the `[text, image]` sequence: text tokens sit at the
        origin, image tokens carry their `(0, h, w)` latent-grid coordinates."""
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

    # ------------------------------------------------------------------
    # LoRA loading (Ostris AI-Toolkit / ComfyUI / diffusers formats)
    # ------------------------------------------------------------------
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: Optional[str] = None,
        adapter_name: str = "default",
        **kwargs,
    ):
        r"""
        Load a Krea 2 LoRA into the transformer.

        Accepts a state dict, a local `.safetensors` file or directory, or a hub repo id
        (with `weight_name` selecting the file when the repo holds several). Handles
        Ostris AI-Toolkit / ComfyUI key layouts (`diffusion_model.blocks...` with
        `lora_A`/`lora_B` or `lora_down`/`lora_up`) as well as already-converted
        diffusers-format state dicts (`transformer.transformer_blocks...`).
        """
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = dict(pretrained_model_name_or_path_or_dict)
        else:
            from safetensors.torch import load_file

            path = str(pretrained_model_name_or_path_or_dict)
            if os.path.isfile(path):
                file_path = path
            elif os.path.isdir(path):
                if weight_name is None:
                    candidates = [f for f in os.listdir(path) if f.endswith(".safetensors")]
                    if len(candidates) != 1:
                        raise ValueError(
                            f"Could not pick a LoRA file in {path}: found {candidates}. Pass `weight_name`."
                        )
                    weight_name = candidates[0]
                file_path = os.path.join(path, weight_name)
            else:
                from huggingface_hub import hf_hub_download, list_repo_files

                if weight_name is None:
                    candidates = [
                        f for f in list_repo_files(path, token=kwargs.get("token", None)) if f.endswith(".safetensors")
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            f"Could not pick a LoRA file in hub repo {path}: found {candidates}. Pass `weight_name`."
                        )
                    weight_name = candidates[0]
                file_path = hf_hub_download(path, weight_name, token=kwargs.get("token", None))
            state_dict = load_file(file_path)

        state_dict = _normalize_lora_state_dict(state_dict)
        if not any(k.startswith("transformer.") for k in state_dict):
            state_dict = _convert_non_diffusers_krea2_lora_to_diffusers(state_dict)

        self.transformer.load_lora_adapter(state_dict, prefix="transformer", adapter_name=adapter_name)

    def unload_lora_weights(self):
        """Remove all loaded LoRA adapters from the transformer."""
        transformer = self.transformer
        if hasattr(transformer, "unload_lora"):
            transformer.unload_lora()
        elif getattr(transformer, "peft_config", None):
            transformer.delete_adapters(list(transformer.peft_config.keys()))

    def fuse_lora(self, lora_scale: float = 1.0, adapter_names: Optional[List[str]] = None, **kwargs):
        """Fuse the loaded LoRA weights into the transformer for adapter-free inference."""
        self.transformer.fuse_lora(lora_scale=lora_scale, adapter_names=adapter_names, **kwargs)

    def unfuse_lora(self, **kwargs):
        self.transformer.unfuse_lora(**kwargs)

    def set_adapters(self, adapter_names: Union[str, List[str]], weights: Optional[Union[float, List[float]]] = None):
        """Activate (and optionally weight) specific loaded LoRA adapters."""
        self.transformer.set_adapters(adapter_names, weights)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], None] = None,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List, None] = None,
        negative_prompt: Union[str, List[str], None] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Union[torch.Generator, List[torch.Generator], None] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        reference_max_pixels: int = 1024 * 1024,
        reference_placements: Optional[List[Dict[str, Any]]] = None,
        known_image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, None] = None,
        known_mask: Union[PIL.Image.Image, np.ndarray, torch.Tensor, None] = None,
        vl_image_max_pixels: int = 384 * 384,
        encode_reference_in_prompt: bool = True,
        kv_cache: bool = False,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Generate images from a prompt, optionally conditioned on reference images.

        Args:
            prompt (`str` or `list[str]`):
                The prompt(s) to guide generation. For edits, describe the change (e.g.
                "make the sky purple").
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor` or a list of them, *optional*):
                Reference image(s). They are encoded into the prompt conditioning via
                the Qwen3-VL vision tower and appended to the transformer sequence as
                clean VAE latents at t=0. References keep their own aspect ratio; the
                output size is set by `height`/`width` independently.
            negative_prompt (`str` or `list[str]`, *optional*):
                Prompt(s) not to guide generation; ignored when `guidance_scale <= 0`.
            height / width (`int`, defaults to 1024):
                Output size in pixels; rounded up to a multiple of 16 if needed.
            num_inference_steps (`int`, *optional*):
                Denoising steps. Defaults to 8 for a distilled (Turbo) checkpoint and 28
                otherwise.
            sigmas (`list[float]`, *optional*):
                Custom sigma grid for the scheduler.
            guidance_scale (`float`, *optional*):
                Krea 2 CFG convention: velocity is `cond + scale * (cond - uncond)` and
                guidance is enabled whenever `scale > 0` (equals standard CFG with scale
                `1 + scale`). Defaults to 0.0 for a distilled checkpoint and 4.5
                otherwise.
            num_images_per_prompt (`int`, defaults to 1):
                Number of images per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated packed noisy latents `(B, image_seq_len, in_channels)`.
            prompt_embeds / prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-computed text conditioning `(B, L, num_text_layers, D)` and its
                bool mask `(B, L)`; skips prompt encoding when given.
            negative_prompt_embeds / negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Same, for the negative prompt.
            reference_max_pixels (`int`, defaults to `1024 * 1024`):
                Pixel budget each reference image is downscaled to fit before VAE
                encoding (never upscaled).
            known_image / known_mask (`PIL.Image`, `numpy.ndarray`, or `torch.Tensor`, *optional*):
                A full-canvas clean image and binary keep mask for inpaint/outpaint
                preservation. After every flow step, mask-selected tokens are replaced
                with the clean latent noised to the next scheduler sigma using the same
                initial noise. Supply both or neither.
            vl_image_max_pixels (`int`, defaults to `384 * 384`):
                Pixel budget for the (coarse) Qwen3-VL view of each reference image.
            encode_reference_in_prompt (`bool`, defaults to `True`):
                Whether reference images are also embedded into the text conditioning
                through the Qwen3-VL vision tower (matches AI-Toolkit edit training).
            kv_cache (`bool`, defaults to `False`):
                Cache the reference tokens' attention K/V: they are precomputed
                in a single t=0 pass and reused on every denoising step, so the
                reference tokens never ride along in the per-step sequence --
                faster, especially with CFG or many steps. The LoRA must be
                trained with AI-Toolkit's ``kv_cache`` model kwarg (reference
                tokens attend only to each other) for this to work properly;
                leave off for normally trained edit LoRAs.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`Krea2PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Forwarded to the transformer; a `scale` entry sets the LoRA scale.
            max_sequence_length (`int`, defaults to 512):
                Maximum prompt token length (truncation only; no fixed padding).

        Returns:
            [`Krea2PipelineOutput`] or `tuple`: the generated images.
        """
        if num_inference_steps is None:
            num_inference_steps = 8 if self.config.is_distilled else 28
        if guidance_scale is None:
            guidance_scale = 0.0 if self.config.is_distilled else 4.5

        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            rounded_height = ((height + multiple - 1) // multiple) * multiple
            rounded_width = ((width + multiple - 1) // multiple) * multiple
            logger.warning(
                f"`height` and `width` must be multiples of {multiple}; rounding up from {height}x{width} to"
                f" {rounded_height}x{rounded_width}."
            )
            height, width = rounded_height, rounded_width

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("`prompt_embeds` requires `prompt_embeds_mask`.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("`negative_prompt_embeds` requires `negative_prompt_embeds_mask`.")
        if (known_image is None) != (known_mask is None):
            raise ValueError("`known_image` and `known_mask` must be supplied together.")

        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        # 1. Normalize reference images to a list of [0, 1] CHW tensors.
        ref_images = None
        if image is not None:
            image_list = image if isinstance(image, (list, tuple)) else [image]
            ref_images = [self._to_chw_tensor(img) for img in image_list]

        # 2. Encode the prompt(s). With references, the coarse VL view of each image is
        # embedded in the user message so the text conditioning "sees" them.
        vl_images = None
        if ref_images is not None and encode_reference_in_prompt:
            vl_images = self._prep_vl_images([img.to(device) for img in ref_images], vl_image_max_pixels)

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                prompt, vl_images, num_images_per_prompt, max_sequence_length, device
            )
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt = negative_prompt if negative_prompt is not None else ""
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * batch_size
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    negative_prompt, vl_images, num_images_per_prompt, max_sequence_length, device
                )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Prepare the noisy latents (kept in float32 across scheduler steps).
        num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        grid_height = height // (self.vae_scale_factor * self.patch_size)
        grid_width = width // (self.vae_scale_factor * self.patch_size)
        initial_noise = latents.clone() if known_image is not None else None

        known_latents = None
        known_token_mask = None
        if known_image is not None:
            known_tensor = self._to_chw_tensor(known_image)
            if tuple(known_tensor.shape[-2:]) != (height, width):
                raise ValueError("`known_image` must match the requested output dimensions.")
            known_latent = self._encode_reference_latents(
                [known_tensor], height * width, generator, device
            )[0].unsqueeze(0)
            known_latents = self._pack_latents(known_latent).float()
            known_latents = known_latents.expand(latents.shape[0], -1, -1)

            if isinstance(known_mask, torch.Tensor):
                mask_tensor = known_mask.detach().float()
                while mask_tensor.ndim > 2:
                    mask_tensor = mask_tensor[0]
                if mask_tensor.max() > 1:
                    mask_tensor = mask_tensor / 255.0
            else:
                if isinstance(known_mask, np.ndarray):
                    mask_array = known_mask
                else:
                    mask_array = np.asarray(known_mask.convert("L"))
                if mask_array.ndim == 3:
                    mask_array = mask_array[..., 0]
                mask_tensor = torch.from_numpy(np.asarray(mask_array).copy()).float() / 255.0
            if tuple(mask_tensor.shape) != (height, width):
                raise ValueError("`known_mask` must match the requested output dimensions.")
            latent_h = height // self.vae_scale_factor
            latent_w = width // self.vae_scale_factor
            mask_tensor = F.interpolate(
                mask_tensor[None, None].to(device),
                size=(latent_h, latent_w),
                mode="nearest",
            )
            p = self.patch_size
            mask_tensor = mask_tensor.view(1, 1, latent_h // p, p, latent_w // p, p)
            known_token_mask = (mask_tensor.amin(dim=(1, 3, 5)) > 0.5).reshape(1, -1, 1)
            known_token_mask = known_token_mask.expand(latents.shape[0], -1, -1)

        # 4. Encode + pack reference latents (shared across the batch) and build the
        # combined rotary coordinates.
        ref_tokens, ref_seq_len = None, 0
        neg_position_ids = None
        position_ids = self.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)
        if self.do_classifier_free_guidance:
            neg_position_ids = self.prepare_position_ids(
                negative_prompt_embeds.shape[1], grid_height, grid_width, device
            )
        ref_kv = None
        if ref_images is not None:
            if reference_placements is not None and len(reference_placements) != len(ref_images):
                raise ValueError("`reference_placements` must match the number of reference images.")
            ref_latents = self._encode_reference_latents(ref_images, reference_max_pixels, generator, device)
            ref_tokens, ref_position_ids = self._pack_reference_latents(
                ref_latents,
                device,
                transformer_dtype,
                placements=reference_placements,
                target_grid_size=(grid_height, grid_width),
            )
            ref_seq_len = ref_tokens.shape[1]
            ref_tokens = ref_tokens.expand(latents.shape[0], -1, -1)
            if kv_cache:
                # Precompute pass: the refs alone run through the blocks once at t=0
                # and every denoising step reuses their per-block K/V, so the ref
                # tokens are dropped from the per-step sequence entirely.
                ref_kv = self.transformer.precompute_ref_kv(ref_tokens, ref_position_ids, attention_kwargs)
                ref_tokens, ref_seq_len = None, 0
            else:
                position_ids = torch.cat([position_ids, ref_position_ids], dim=0)
                if neg_position_ids is not None:
                    neg_position_ids = torch.cat([neg_position_ids, ref_position_ids], dim=0)

        # 5. Prepare timesteps. The distilled (Turbo) checkpoint was trained at a fixed
        # exponential time shift mu=1.15; the base checkpoint interpolates mu from the
        # image token count.
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if self.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                grid_height * grid_width,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # 6. Denoising loop (Euler flow ODE integration via the scheduler).
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, t in enumerate(timesteps):
                timestep = (t / self.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(
                    transformer_dtype
                )

                model_input = latents.to(transformer_dtype)
                if ref_tokens is not None:
                    model_input = torch.cat([model_input, ref_tokens], dim=1)

                noise_pred = self.transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_embeds_mask,
                    ref_seq_len=ref_seq_len,
                    ref_kv_cache=ref_kv,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    neg_noise_pred = self.transformer(
                        hidden_states=model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep,
                        position_ids=neg_position_ids,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        ref_seq_len=ref_seq_len,
                        ref_kv_cache=ref_kv,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]
                if known_latents is not None:
                    next_sigma = self.scheduler.sigmas[step_index + 1].to(
                        device=latents.device, dtype=latents.dtype
                    )
                    noised_known = known_latents + next_sigma * (
                        initial_noise - known_latents
                    )
                    latents = torch.where(known_token_mask, noised_known, latents)
                progress_bar.update()

        # 7. Decode latents.
        if output_type == "latent":
            image_out = latents
        else:
            latents = self._unpack_latents(latents, height, width).to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return Krea2PipelineOutput(images=image_out)
