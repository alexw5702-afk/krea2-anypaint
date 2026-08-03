# Krea 2 AnyPaint

Arbitrary-mask inpainting, outpainting, and image editing for Krea 2 Turbo.

This repository contains the portable Diffusers runtime for
[`yijunwang2/krea2-anypaint`](https://huggingface.co/yijunwang2/krea2-anypaint).
The model repository provides the rank-32 LoRA weights, license, fixed-seed
showcase, and complete model card. A ZeroGPU demo is available at
[`yijunwang2/krea2-anypaint`](https://huggingface.co/spaces/yijunwang2/krea2-anypaint).

## Features

- arbitrary-shape inpainting masks;
- one-pass outpainting into a larger canvas;
- mixed internal and external image editing;
- disconnected and narrow generated regions;
- timestep-matched preservation of known latent tokens;
- isolated reference K/V caching for distilled Krea 2 Turbo inference.

White mask pixels are generated or edited. Black pixels are preserved. Pixels
outside the optional source bounding box are automatically generated.

## Install

```bash
git clone https://github.com/alexw5702-afk/krea2-anypaint
cd krea2-anypaint
pip install -r requirements.txt
```

## Run

```bash
python example.py \
  --source input.webp \
  --mask mask.png \
  --output result.webp \
  --width 1536 \
  --height 1024 \
  --bbox 0 0 1024 1024 \
  --prompt "a complete coherent scene"
```

The example downloads the adapter from Hugging Face. Access to the Krea 2 base
model is governed by the Krea 2 Community License.

## ComfyUI

ComfyUI support is planned. The existing registered-outpaint nodes can supply
the reference-attention and placement core, but AnyPaint additionally requires
arbitrary mask input and timestep-matched known-region replacement in the
sampling loop. Loading the AnyPaint LoRA into the old rectangular workflow is
not sufficient.

## Scope

This runtime repository does not contain training data, training caches,
optimizer state, or intermediate checkpoints. The bundled showcase in the
model repository uses only synthetic, original, non-IP source images.

## License

Runtime code is distributed under Apache License 2.0. The separately downloaded
LoRA weights are a Krea 2 derivative and remain subject to the Krea 2 Community
License. See `NOTICE` and the model repository for details.
