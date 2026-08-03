# ComfyUI Krea 2 AnyPaint

Native ComfyUI nodes for the
[`yijunwang2/krea2-anypaint`](https://huggingface.co/yijunwang2/krea2-anypaint)
functional adapter. AnyPaint adds arbitrary-mask **inpainting**, **outpainting**,
and mixed **image editing** to Krea 2 Turbo.

[Model and portable Diffusers example](https://huggingface.co/yijunwang2/krea2-anypaint)
| [Interactive ZeroGPU Space](https://huggingface.co/spaces/yijunwang2/krea2-anypaint)
| [Collection](https://huggingface.co/collections/yijunwang2/krea-2-functional-adapters-6a700e9e5c134888d6615a5d)

![AnyPaint examples](showcase.webp)

<!-- SHOWCASE_EXAMPLES_START -->
<details>
<summary><strong>Open the complete source / mask / result set and prompts</strong></summary>

### 01_photo_internal - Photographic internal edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/01_photo_internal_source.webp) | ![](showcase/01_photo_internal_mask.png) | ![](showcase/01_photo_internal_result.webp) |

**Prompt:** a sunlit modern kitchen interior with a handmade ceramic bowl of fresh lemons on the center island, coherent morning light, complete architectural photograph

### 02_watercolor_outpaint - Watercolor downward outpainting

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/02_watercolor_outpaint_source.webp) | ![](showcase/02_watercolor_outpaint_mask.png) | ![](showcase/02_watercolor_outpaint_result.webp) |

**Prompt:** an original watercolor greenhouse and coastal garden, the path and flowering plants continuing naturally downward, airy washes, complete vertical composition

### 03_3d_outpaint - Stylized 3D horizontal outpainting

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/03_3d_outpaint_source.webp) | ![](showcase/03_3d_outpaint_mask.png) | ![](showcase/03_3d_outpaint_result.webp) |

**Prompt:** an original retro-futuristic exploration rover crossing a colorful alien desert, distant mineral formations extending to the right, stylized 3D cinematic render

### 04_portrait_mixed - Original-character mixed edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/04_portrait_mixed_source.webp) | ![](showcase/04_portrait_mixed_mask.png) | ![](showcase/04_portrait_mixed_result.webp) |

**Prompt:** an original adult woman with short silver hair wearing a richly textured coral coat in a spacious contemporary gallery, complete full-body fashion photograph

### 05_botanical_disconnected - Disconnected honeybee insertion

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/05_botanical_disconnected_source.webp) | ![](showcase/05_botanical_disconnected_mask.png) | ![](showcase/05_botanical_disconnected_result.webp) |

**Prompt:** an original botanical field illustration of blue and white meadow flowers with three small honeybees hovering naturally between the blossoms, delicate wings, fine paper texture, naturalistic painted details

### 07_product_narrow - Narrow cable recolor

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/07_product_narrow_source.webp) | ![](showcase/07_product_narrow_mask.png) | ![](showcase/07_product_narrow_result.webp) |

**Prompt:** an original sculptural brushed aluminum desk lamp with a vivid orange braided power cable extending from the lamp across the tabletop, premium neutral product photography

### 08_landscape_complex - Complex semantic landscape edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/08_landscape_complex_source.webp) | ![](showcase/08_landscape_complex_mask.png) | ![](showcase/08_landscape_complex_result.webp) |

**Prompt:** an original alpine valley after rain with clusters of luminous rainbow-colored fantasy mushrooms growing along the winding riverbanks and foreground rocks, subtle magical glow reflected in wet stone, coherent wide cinematic landscape photograph

### 09_male_accessories - Male portrait accessories

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/09_male_accessories_source.webp) | ![](showcase/09_male_accessories_mask.png) | ![](showcase/09_male_accessories_result.webp) |

**Prompt:** a realistic editorial portrait of an adult man with short dark hair wearing thin rectangular silver eyeglasses and a small silver ear cuff, plain navy shirt, coherent soft daylight and neutral studio background

### 10_female_accessories - Female disconnected accessories

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/10_female_accessories_source.webp) | ![](showcase/10_female_accessories_mask.png) | ![](showcase/10_female_accessories_result.webp) |

**Prompt:** a polished portrait of an adult woman with shoulder-length auburn hair, a dark green enamel hair clip above her left ear and matching small geometric earrings, simple cream blouse, coherent natural daylight and pale gray studio background

### 11_product_restyle - Shoes and bag local redesign

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/11_product_restyle_source.webp) | ![](showcase/11_product_restyle_mask.png) | ![](showcase/11_product_restyle_result.webp) |

**Prompt:** an overhead catalog photograph with a pair of deep forest-green suede sneakers on the left and a cobalt-blue structured leather shoulder bag on the right, refined minimal hardware, balanced composition, coherent soft studio shadows

### 12_desk_objects - Two desktop object insertions

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/12_desk_objects_source.webp) | ![](showcase/12_desk_objects_mask.png) | ![](showcase/12_desk_objects_result.webp) |

**Prompt:** a clean oak writing desk in a softly lit home office with a handmade dark green ceramic mug on the left side and a small potted succulent in a matte white planter on the right, coherent restrained contemporary interior photograph

### 13_living_room_objects - Living-room multi-region insertion

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/13_living_room_objects_source.webp) | ![](showcase/13_living_room_objects_mask.png) | ![](showcase/13_living_room_objects_result.webp) |

**Prompt:** a bright neutral living room with a tall brushed-brass floor lamp beside the left end of the cream sofa and a round woven storage basket beside the right end, simple coffee table, coherent warm daylight and understated architectural photography

### 14_window_view - Complete window-view replacement

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/14_window_view_source.webp) | ![](showcase/14_window_view_mask.png) | ![](showcase/14_window_view_result.webp) |

**Prompt:** a quiet contemporary reading room whose large central picture window overlooks a snowy mountain lake beneath a clear pale-blue winter sky, armchair in the foreground, coherent daylight, reflections and realistic interior photography

### 15_person_boundary - Boundary completion of an adult man

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/15_person_boundary_source.webp) | ![](showcase/15_person_boundary_mask.png) | ![](showcase/15_person_boundary_result.webp) |

**Prompt:** a complete full-body photograph of an adult man wearing a charcoal field jacket, gray shirt, straight dark trousers and practical brown boots, standing naturally on a quiet railway platform in coherent soft cloudy daylight

</details>
<!-- SHOWCASE_EXAMPLES_END -->

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexw5702-afk/krea2-anypaint
```

Restart ComfyUI, then place
`krea2_anypaint_rank32.safetensors` in `ComfyUI/models/loras`.

## Requirements

- A current ComfyUI build with native Krea 2 support.
- [`krea2_turbo_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_int8_convrot.safetensors)
  in `ComfyUI/models/diffusion_models`.
- [`qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors)
  in `ComfyUI/models/text_encoders`.
- [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors)
  in `ComfyUI/models/vae`.
- [Krea 2 AnyPaint LoRA](https://huggingface.co/yijunwang2/krea2-anypaint/blob/main/krea2_anypaint_rank32.safetensors).

No custom sampler and no final source composite are required.

## Nodes

### Krea2 AnyPaint Prepare

Takes a source image, optional arbitrary mask, and four padding values. White
mask pixels and all padded canvas pixels are generated; black pixels are
preserved. The node outputs:

- a 384px semantic reference for Krea 2 reference attention and Qwen3-VL;
- the full-resolution known canvas;
- generated and preserved masks;
- aligned canvas dimensions.

Use the `MASK` output from ComfyUI's `Load Image` Mask Editor directly. For
inpainting, keep all padding at zero. For outpainting, add space on any side.
Both can be combined in one request.

### Krea2 AnyPaint Encode

Encodes the prompt/reference and returns a standard ComfyUI `LATENT` containing
the known image plus a token-aligned `noise_mask`. ComfyUI's sampler applies the
same timestep-matched preservation used by the Diffusers reference pipeline:

```text
known_at_sigma = sigma * initial_noise + (1 - sigma) * known_latent
```

### Krea2 AnyPaint Model Patch

Registers the semantic reference over the complete target grid and precomputes
its isolated reference K/V cache. Connect this node after `Load LoRA` and before
`KSampler`.

## Workflow

Drag [`krea2_anypaint_workflow.json`](krea2_anypaint_workflow.json) into
ComfyUI. The important wiring is:

```text
Load Image (image + mask) -> AnyPaint Prepare
AnyPaint Prepare + CLIP + VAE -> AnyPaint Encode
Load Diffusion Model -> Load LoRA -> AnyPaint Model Patch -> KSampler
AnyPaint Encode.latent -> KSampler.latent_image
KSampler -> VAE Decode -> Save Image
```

Recommended Krea 2 Turbo settings: 8 steps, Euler, simple scheduler, CFG 1.0,
LoRA strength 1.0, VLM reference enabled, K/V cache enabled.

This graph was validated end to end with the three linked Comfy-Org model
files. The preprocessing, mask convention, reference placement, sigma schedule,
and per-step known-region replacement match the Diffusers example and the
DeepMachine backend. Quantized runtimes may differ at the pixel level from the
portable BF16 example.

The output is the raw model decode. Do not add the old registered-outpaint
composite node: AnyPaint preserves known pixels during every denoising step.

## Credits

Reference-attention and K/V-cache code is adapted from
[`RealRebelAI/ComfyUI-Rebels-Krea2-Outpaint`](https://github.com/RealRebelAI/ComfyUI-Rebels-Krea2-Outpaint)
and [`ostris/ComfyUI-Krea2-Ostris-Edit`](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit).
See `NOTICE` for the complete attribution chain.

Node code is MIT licensed. The separately downloaded LoRA is a Krea 2
derivative and remains subject to the Krea 2 Community License.
