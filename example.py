from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from anypaint import prepare_anypaint


REPO_ID = "yijunwang2/krea2-anypaint"
WEIGHT_NAME = "krea2_anypaint_rank32.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run arbitrary-mask Krea 2 inpainting and outpainting"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Source placement in output pixels; defaults to the full canvas",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_anypaint(
        Image.open(args.source),
        Image.open(args.mask),
        (args.width, args.height),
        tuple(args.bbox) if args.bbox else None,
    )

    pipe = DiffusionPipeline.from_pretrained(
        "krea/Krea-2-Turbo",
        custom_pipeline=REPO_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(REPO_ID, weight_name=WEIGHT_NAME, adapter_name="anypaint")
    pipe.set_adapters(["anypaint"], adapter_weights=[args.lora_scale])

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        image=prepared.condition,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
        reference_max_pixels=384 * 384,
        reference_placements=[prepared.reference_placement],
        vlm_reference_indices=[0],
        encode_reference_in_prompt=False,
        kv_cache=True,
        known_image=prepared.known_image,
        known_mask=prepared.keep_mask,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == "__main__":
    main()
