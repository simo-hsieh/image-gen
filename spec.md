# Photo → Cartoon Style Generator with ComfyUI

**Goal:** Build a Python script that converts a personal photo into cartoon/anime/Ghibli style images using ComfyUI's API. Generate multiple variations.

---

## Environment Setup

- Install ComfyUI: `pip install comfy-cli` then `comfy install`
- Python env needs: `websocket-client`
- Run server: `python main.py` inside ComfyUI folder (uses MPS on Apple Silicon automatically)
- Server runs at `http://127.0.0.1:8188`

---

## Models to Download

> via `comfy model download --url <url> --relative-path <path>`

| Model | URL | Path |
|-------|-----|------|
| Checkpoint | <https://huggingface.co/frankjoshua/toonyou_beta6/resolve/main/toonyou_beta6.safetensors> | models/checkpoints |
| ControlNet | <https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth> | models/controlnet |
| Ghibli LoRA | <https://huggingface.co/Lykon/LoRA/resolve/main/ghibli_style_offset.safetensors> | models/loras |

---

## Script: `generate_versions.py`

**Output naming:** `<source-stem>_<flow>_d<denoise*100>_cn<cn*10>_s<seed>.png`  
**Example:** `5229-0-2_ghibli_d050_cn08_s42.png`

### Flow

For each combination of (flow, denoise, controlnet_strength, seed):

1. Upload image to `/upload/image` (multipart POST) → get filename
2. Build workflow JSON (10 nodes + optional LoRA node 11)
3. POST to `/prompt` with `client_id` + `prompt_id`
4. Connect WebSocket `ws://127.0.0.1:8188/ws?clientId=<id>`, wait for `executing → node: null`
5. Fetch `/history/<prompt_id>` → get output filenames → fetch `/view?filename=...`
6. Save to disk with systematic name

---

## 10-Node Workflow (img2img + ControlNet)

1. `CheckpointLoaderSimple` (toonyou_beta6.safetensors)
2. `LoadImage` (uploaded photo)
3. `VAEEncode` (pixels=2.0, vae=1.2) → latent starting point
4. `[11. LoraLoader (optional, model=1.0, clip=1.1, strength=0.85)]` → patches model+clip
5. `CLIPTextEncode` positive (clip=1.1 or 11.1)
6. `CLIPTextEncode` negative (clip=1.1 or 11.1)
7. `ControlNetLoader` (control_v11p_sd15_lineart.pth)
8. `ControlNetApplyAdvanced` (pos=4, neg=5, cn=6, image=2, start=0.0, end=1.0)
9. `KSampler` (model=1.0/11.0, pos=7.0, neg=7.1, latent=3.0, steps=25, cfg=7, sampler=dpmpp_2m, scheduler=karras)
10. `VAEDecode` (samples=8.0, vae=1.2)
11. `SaveImage` (images=9.0, prefix="cartoon")

---

## Tuning Parameters

- **denoise** `0.4–0.55` = stays close to your face; `0.7+` = heavy style transfer
- **controlnet_strength** `0.85–0.95` = preserves face structure; `end_percent=1.0` = guides full generation
- **LoRA** `strength_model/clip = 0.85` for Ghibli style influence

---

## Flows Defined

| Flow | Positive Keywords | Negative Keywords |
|------|-------------------|-------------------|
| cartoon | cartoon style, soft colors, gentle cel shading, yellow hair, blonde hair, soft rounded chin | sharp chin, angular jaw, dark hair, realistic |
| pixar | pixar style, 3d render, soft lighting, yellow hair, soft rounded chin | sharp chin, dark hair, flat |
| anime | anime style, pastel colors, studio ghibli, yellow hair, soft rounded chin | sharp chin, dark hair, western cartoon |
| ghibli | ghibli style, miyazaki, soft watercolor, yellow hair, soft rounded chin | sharp chin, dark hair, 3d render |

---

## CLI Usage

```bash
python generate_versions.py --image photo.jpg \
  --flow ghibli cartoon \
  --denoise 0.45 0.5 0.55 \
  --controlnet-strength 0.85 \
  --seeds 42 125 456 \
  --output-dir ./versions
```
