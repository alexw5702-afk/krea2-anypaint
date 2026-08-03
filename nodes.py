"""ComfyUI nodes for the yijunwang2 Krea 2 AnyPaint adapter."""

from __future__ import annotations

import math

import torch

import comfy.conds
import node_helpers

from . import anypaint_helpers as ah
from . import krea2_registered_core as core


class Krea2AnyPaintPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "right": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 16}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "reference_max_edge": ("INT", {"default": 384, "min": 128, "max": 768, "step": 16}),
                "boundary_redraw_px": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
            },
            "optional": {
                "generated_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT")
    RETURN_NAMES = (
        "semantic_reference",
        "known_image",
        "generated_mask",
        "keep_mask",
        "canvas_width",
        "canvas_height",
    )
    FUNCTION = "prepare"
    CATEGORY = "Krea2/AnyPaint"
    DESCRIPTION = (
        "Place the source without resizing it. Padding and white mask pixels are "
        "generated; black mask pixels are preserved. The source image's Mask "
        "Editor output can be connected directly to generated_mask."
    )

    def prepare(
        self,
        source,
        left,
        top,
        right,
        bottom,
        reference_max_edge,
        boundary_redraw_px,
        generated_mask=None,
    ):
        if generated_mask is None:
            generated_mask = torch.zeros(
                (source.shape[0], source.shape[1], source.shape[2]),
                dtype=torch.float32,
                device=source.device,
            )
        return ah.prepare_canvas(
            source,
            generated_mask,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            reference_max_edge=reference_max_edge,
            seam_px=boundary_redraw_px,
        )


def _encode_reference_latent(vae, image):
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    target_h = ah.align_up(height)
    target_w = ah.align_up(width)
    if (target_h, target_w) != (height, width):
        samples = torch.nn.functional.interpolate(
            samples,
            (target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    return vae.encode(samples.movedim(1, -1)[..., :3])


class Krea2AnyPaintEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vae": ("VAE",),
                "semantic_reference": ("IMAGE",),
                "known_image": ("IMAGE",),
                "keep_mask": ("MASK",),
                "vlm_reference": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "Krea2/AnyPaint"
    DESCRIPTION = (
        "Encode the full-canvas semantic reference and create a standard Comfy "
        "latent/noise_mask pair. KSampler performs timestep-matched known-region "
        "replacement; no final source composite is used."
    )

    def encode(
        self,
        clip,
        prompt,
        vae,
        semantic_reference,
        known_image,
        keep_mask,
        vlm_reference=True,
    ):
        images_vl = [semantic_reference] if vlm_reference else []
        image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" if images_vl else ""
        if images_vl:
            try:
                from comfy.text_encoders.krea2 import KREA2_TEMPLATE

                tokens = clip.tokenize(
                    image_prompt + prompt,
                    images=images_vl,
                    llama_template=KREA2_TEMPLATE,
                )
            except Exception:
                tokens = clip.tokenize(image_prompt + prompt, images=images_vl)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        reference_latent = _encode_reference_latent(vae, semantic_reference)
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [reference_latent]},
            append=True,
        )

        known_latent = vae.encode(known_image[..., :3])
        if known_latent.ndim not in (4, 5):
            raise ValueError(f"AnyPaint expects a 4D or 5D image latent, got {tuple(known_latent.shape)}")
        if known_latent.ndim == 5 and known_latent.shape[2] != 1:
            raise ValueError(
                "AnyPaint handles still images only; the latent time dimension must be 1, "
                f"got {tuple(known_latent.shape)}"
            )
        noise_mask = ah.token_aligned_noise_mask(
            keep_mask,
            known_latent.shape[-2],
            known_latent.shape[-1],
            patch_size=2,
        )
        return conditioning, {"samples": known_latent, "noise_mask": noise_mask}


class Krea2AnyPaintModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Krea2/AnyPaint"
    DESCRIPTION = (
        "Register the semantic reference over the complete target canvas and "
        "cache its isolated K/V once per sampling run. Place after Load LoRA."
    )

    def patch(self, model, kv_cache=True):
        bbox_norm = [0.0, 0.0, 1.0, 1.0]
        patched = model.clone()
        base_model = patched.model
        dit = patched.get_model_object("diffusion_model")

        original_extra_conds = base_model.extra_conds
        original_extra_conds_shapes = base_model.extra_conds_shapes
        original_forward = dit.forward

        def extra_conds(**kwargs):
            output = original_extra_conds(**kwargs)
            reference_latents = kwargs.get("reference_latents")
            if reference_latents is not None:
                output["ref_latents"] = comfy.conds.CONDList(
                    [base_model.process_latent_in(latent) for latent in reference_latents]
                )
            return output

        def extra_conds_shapes(**kwargs):
            output = original_extra_conds_shapes(**kwargs)
            reference_latents = kwargs.get("reference_latents")
            if reference_latents is not None:
                output["ref_latents"] = [
                    1,
                    16,
                    sum(math.prod(latent.size()) for latent in reference_latents) // 16,
                ]
            return output

        state = {"last_sigma": None, "caches": {}}

        def forward(
            x,
            timesteps,
            context,
            attention_mask=None,
            transformer_options={},
            ref_latents=None,
            **kwargs,
        ):
            if not ref_latents:
                return original_forward(
                    x,
                    timesteps,
                    context,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            if not kv_cache:
                return core._forward_with_refs(
                    dit,
                    x,
                    timesteps,
                    context,
                    ref_latents,
                    transformer_options,
                    bbox_norm=bbox_norm,
                )

            sigma = float(timesteps.max())
            sample_sigmas = transformer_options.get("sample_sigmas")
            new_run = state["last_sigma"] is None or sigma > state["last_sigma"]
            if sample_sigmas is not None and sigma == float(sample_sigmas[0]) and sigma != state["last_sigma"]:
                new_run = True
            if new_run:
                state["caches"].clear()
            state["last_sigma"] = sigma

            batch_size = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
            key = core._ref_fingerprint(ref_latents, batch_size, bbox_norm)
            reference_kv = state["caches"].get(key)
            if reference_kv is None:
                reference_kv = core._precompute_ref_kv(
                    dit,
                    x,
                    timesteps,
                    ref_latents,
                    transformer_options,
                    bbox_norm=bbox_norm,
                )
                state["caches"][key] = reference_kv
            return core._forward_with_cached_refs(
                dit,
                x,
                timesteps,
                context,
                reference_kv,
                transformer_options,
            )

        patched.add_object_patch("extra_conds", extra_conds)
        patched.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        patched.add_object_patch("diffusion_model.forward", forward)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "Krea2AnyPaintPrepare": Krea2AnyPaintPrepare,
    "Krea2AnyPaintEncode": Krea2AnyPaintEncode,
    "Krea2AnyPaintModelPatch": Krea2AnyPaintModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2AnyPaintPrepare": "Krea2 AnyPaint Prepare",
    "Krea2AnyPaintEncode": "Krea2 AnyPaint Encode",
    "Krea2AnyPaintModelPatch": "Krea2 AnyPaint Model Patch",
}
