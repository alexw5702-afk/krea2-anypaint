from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


REFERENCE_MAX_EDGE = 384
SEAM_PX = 32


@dataclass(frozen=True)
class AnyPaintInput:
    condition: Image.Image
    known_image: Image.Image
    keep_mask: Image.Image
    generated_mask: Image.Image
    canvas_size: tuple[int, int]
    source_bbox: tuple[int, int, int, int]

    @property
    def reference_placement(self) -> dict[str, list[float]]:
        return {"bbox_normalized": [0.0, 0.0, 1.0, 1.0]}


def _validate_canvas(canvas_size: tuple[int, int]) -> None:
    width, height = canvas_size
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image.copy()
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _median_color(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([127, 127, 127], dtype=np.uint8)
    return np.median(values.reshape(-1, 3), axis=0).round().astype(np.uint8)


def _edge_aware_keep_mask(generated_mask: Image.Image, seam_px: int) -> Image.Image:
    generated = np.where(
        np.asarray(generated_mask.convert("L")) > 0,
        255,
        0,
    ).astype(np.uint8)
    if seam_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (seam_px * 2 + 1, seam_px * 2 + 1),
        )
        generated = cv2.dilate(generated, kernel)
    return ImageOps.invert(Image.fromarray(generated, mode="L"))


def prepare_anypaint(
    source: Image.Image,
    generated_mask: Image.Image,
    canvas_size: tuple[int, int],
    source_bbox: tuple[int, int, int, int] | None = None,
    *,
    reference_max_edge: int = REFERENCE_MAX_EDGE,
    seam_px: int = SEAM_PX,
) -> AnyPaintInput:
    """Prepare one arbitrary-mask inpainting/outpainting request.

    White mask pixels are generated. Black mask pixels are preserved. Pixels
    outside ``source_bbox`` are always generated, which makes the same helper
    work for both inpainting and outpainting.
    """
    _validate_canvas(canvas_size)
    width, height = canvas_size
    source = source.convert("RGB")
    if source_bbox is None:
        source_bbox = (0, 0, width, height)
    x0, y0, x1, y1 = source_bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {source_bbox}")

    box_width, box_height = x1 - x0, y1 - y0
    source_ratio = source.width / source.height
    box_ratio = box_width / box_height
    tolerance = max(0.025, 2.0 / min(box_width, box_height))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")

    placed = source.resize((box_width, box_height), Image.Resampling.LANCZOS)
    placed_values = np.asarray(placed, dtype=np.uint8)
    fill = _median_color(placed_values)
    known_values = np.empty((height, width, 3), dtype=np.uint8)
    known_values[:] = fill
    known_values[y0:y1, x0:x1] = placed_values
    known_image = Image.fromarray(known_values, mode="RGB")

    mask = generated_mask.convert("L")
    if mask.size == source.size:
        mask = mask.resize((box_width, box_height), Image.Resampling.NEAREST)
        canvas_mask = Image.new("L", canvas_size, 255)
        canvas_mask.paste(mask, (x0, y0))
        mask = canvas_mask
    elif mask.size != canvas_size:
        raise ValueError("Mask must match either the source image or the output canvas")
    mask = mask.point(lambda value: 255 if value > 127 else 0)

    outside = Image.new("L", canvas_size, 255)
    outside.paste(0, source_bbox)
    generated = Image.fromarray(
        np.maximum(np.asarray(mask, dtype=np.uint8), np.asarray(outside, dtype=np.uint8)),
        mode="L",
    )
    generated_values = np.asarray(generated, dtype=np.uint8) > 0
    if not generated_values.any():
        raise ValueError("The generated mask has no white pixels")

    condition_values = known_values.copy()
    known_pixels = condition_values[~generated_values]
    condition_values[generated_values] = _median_color(known_pixels)
    condition = _resize_max_edge(
        Image.fromarray(condition_values, mode="RGB"),
        reference_max_edge,
    )

    return AnyPaintInput(
        condition=condition,
        known_image=known_image,
        keep_mask=_edge_aware_keep_mask(generated, seam_px),
        generated_mask=generated,
        canvas_size=canvas_size,
        source_bbox=source_bbox,
    )
