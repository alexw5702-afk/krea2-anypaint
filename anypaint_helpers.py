from __future__ import annotations

import math

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


def _resize_reference(image: torch.Tensor, max_edge: int) -> torch.Tensor:
    _, height, width, _ = image.shape
    scale = min(1.0, max_edge / max(height, width))
    target_h = max(ALIGNMENT, int(round(height * scale / ALIGNMENT)) * ALIGNMENT)
    target_w = max(ALIGNMENT, int(round(width * scale / ALIGNMENT)) * ALIGNMENT)
    if (target_h, target_w) == (height, width):
        return image
    chw = image.movedim(-1, 1)
    return F.interpolate(chw, (target_h, target_w), mode="bilinear", align_corners=False).movedim(1, -1)


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
        pixels = source[index]
        median = pixels.reshape(-1, 3).median(dim=0).values
        known = median.view(1, 1, 3).expand(canvas_h, canvas_w, 3).clone()
        known[top : top + source_h, left : left + source_w] = pixels

        generated = torch.ones((canvas_h, canvas_w), dtype=torch.float32, device=source.device)
        generated[top : top + source_h, left : left + source_w] = source_mask[index]
        if not bool(generated.any()):
            raise ValueError("the generated mask has no white pixels")

        condition = known.clone()
        preserved = condition[generated < 0.5]
        fill = preserved.median(dim=0).values if preserved.numel() else median
        condition[generated > 0.5] = fill
        known_images.append(known)
        conditions.append(condition)
        generated_masks.append(generated)

    known_image = torch.stack(known_images)
    condition = _resize_reference(torch.stack(conditions), reference_max_edge)
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
