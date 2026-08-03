from __future__ import annotations

import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ALIGNMENT = 16
REFERENCE_MAX_EDGE = 384
SEAM_PX = 32


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return max(alignment, int(math.ceil(value / alignment) * alignment))


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    while mask.ndim > 3:
        mask = mask[:, 0]
    if mask.ndim == 2:
        mask = mask[None]
    return F.interpolate(mask[:, None].float(), (height, width), mode="nearest")[:, 0]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    values = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(values[..., :3], mode="RGB")


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    values = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(values).to(device=device, dtype=torch.float32).div_(255.0)


def _resize_reference(image: Image.Image, max_edge: int) -> Image.Image:
    scale = min(1.0, max_edge / max(image.size))
    width = max(ALIGNMENT, int(round(image.width * scale)) // ALIGNMENT * ALIGNMENT)
    height = max(ALIGNMENT, int(round(image.height * scale)) // ALIGNMENT * ALIGNMENT)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask[:, None], kernel, stride=1, padding=radius)[:, 0]


def prepare_canvas(
    source: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    reference_max_edge: int = REFERENCE_MAX_EDGE,
    seam_px: int = SEAM_PX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build AnyPaint's full canvas without resizing the source.

    Comfy masks use 1 for generated pixels. Padding is always generated, and
    the supplied mask is placed over the source rectangle for arbitrary
    inpainting. Extra alignment pixels are added on the right/bottom only.
    """
    if source.ndim != 4 or source.shape[-1] < 3:
        raise ValueError("source must be a ComfyUI IMAGE tensor [B,H,W,C]")
    if min(left, top, right, bottom) < 0:
        raise ValueError("padding values must be non-negative")

    source = source[..., :3].detach().float().clamp(0, 1)
    batch, source_h, source_w, _ = source.shape
    canvas_w = align_up(source_w + left + right)
    canvas_h = align_up(source_h + top + bottom)
    source_mask = _resize_mask(generated_mask, source_h, source_w)
    if source_mask.shape[0] == 1 and batch > 1:
        source_mask = source_mask.expand(batch, -1, -1)
    if source_mask.shape[0] != batch:
        raise ValueError("mask batch must be one or match the source batch")
    source_mask = (source_mask > 0.5).float()

    known_images = []
    conditions = []
    generated_masks = []
    for index in range(batch):
        pixels = _tensor_to_pil(source[index])
        pixel_values = np.asarray(pixels, dtype=np.uint8)
        median = np.median(pixel_values.reshape(-1, 3), axis=0).round().astype(np.uint8)
        known_values = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
        known_values[:] = median
        known_values[top : top + source_h, left : left + source_w] = pixel_values

        generated = torch.ones((canvas_h, canvas_w), dtype=torch.float32, device=source.device)
        generated[top : top + source_h, left : left + source_w] = source_mask[index]
        if not bool(generated.any()):
            raise ValueError("the generated mask has no white pixels")

        generated_values = generated.detach().cpu().numpy() > 0.5
        condition_values = known_values.copy()
        preserved = condition_values[~generated_values]
        fill = (
            np.median(preserved, axis=0).round().astype(np.uint8)
            if len(preserved)
            else median
        )
        condition_values[generated_values] = fill
        known_images.append(_pil_to_tensor(Image.fromarray(known_values), source.device))
        conditions.append(
            _pil_to_tensor(
                _resize_reference(Image.fromarray(condition_values), reference_max_edge),
                source.device,
            )
        )
        generated_masks.append(generated)

    known_image = torch.stack(known_images)
    condition = torch.stack(conditions)
    generated = torch.stack(generated_masks)
    keep = 1.0 - _dilate(generated, seam_px).clamp(0, 1)
    return condition, known_image, generated, keep, canvas_w, canvas_h


def token_aligned_noise_mask(
    keep_mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: int = 2,
) -> torch.Tensor:
    """Convert a pixel keep mask to the exact Krea 2 token preservation grid."""
    keep = _resize_mask(keep_mask, latent_height, latent_width)
    keep = (keep > 0.5).float()
    if latent_height % patch_size or latent_width % patch_size:
        raise ValueError("latent dimensions must be divisible by the Krea patch size")
    blocks = keep.view(
        keep.shape[0],
        latent_height // patch_size,
        patch_size,
        latent_width // patch_size,
        patch_size,
    )
    known_blocks = blocks.amin(dim=(2, 4))
    known = known_blocks.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    return (1.0 - known)[:, None]
