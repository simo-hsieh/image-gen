#!/usr/bin/env python3
"""Generate stylized images from photos using Diffusers + ControlNet.

Standalone script — no Ray Serve required. Loads models directly and runs
the same pipeline as serve/serve_diffusion.py.

Usage:
    uv run generate.py --image soruce-images/IMG_7145_1_1_1.JPG --flow ghibli
    uv run generate.py --image soruce-images/IMG_7145_1_1_1.JPG --flow ghibli cartoon --denoise 0.5 0.6
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import time

import torch
from controlnet_aux import LineartDetector
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
from PIL import Image

MODELS_DIR = pathlib.Path(
    os.environ.get("MODELS_DIR", pathlib.Path(__file__).parent / ".." / "ComfyUI" / "models")
)

FLOWS: dict[str, dict[str, str]] = {
    "cartoon": {
        "positive": "cartoon style, soft colors, gentle cel shading, yellow hair, blonde hair, soft rounded chin",
        "negative": "sharp chin, angular jaw, dark hair, realistic",
        "checkpoint": "toonyou_beta6",
    },
    "pixar": {
        "positive": "pixar style, 3d render, soft lighting, yellow hair, soft rounded chin",
        "negative": "sharp chin, dark hair, flat",
        "checkpoint": "toonyou_beta6",
    },
    "anime": {
        "positive": "anime style, pastel colors, studio ghibli, yellow hair, soft rounded chin",
        "negative": "sharp chin, dark hair, western cartoon",
        "checkpoint": "toonyou_beta6",
    },
    "ghibli": {
        "positive": (
            "ghibli style, studio ghibli anime screenshot, miyazaki, soft watercolor, "
            "black hair, glasses, handsome young asian man protagonist, soft facial features, "
            "gentle face, clear smooth skin, kind bright eyes, warm gentle smile, anime hero, "
            "ikemen, bishounen, black jacket, masterpiece, best quality, detailed face, "
            "beautiful warm lighting"
        ),
        "negative": (
            "realistic, 3d render, deformed face, bad anatomy, yellow hair, blonde hair, "
            "low quality, blurry, ugly, aged, wrinkles, dark circles, tired looking, "
            "harsh features, rough skin, angular face, villain"
        ),
        "checkpoint": "ghibli_diffusion",
    },
}

CHECKPOINT_FILES = {
    "toonyou_beta6": "checkpoints/toonyou_beta6.safetensors",
    "ghibli_diffusion": "checkpoints/ghibli_diffusion.ckpt",
}

CONTROLNET_ID = "lllyasviel/control_v11p_sd15_lineart"


def detect_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_pipeline(
    checkpoint_path: str, device: torch.device, dtype: torch.dtype
) -> StableDiffusionControlNetImg2ImgPipeline:
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        checkpoint_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
    )
    if device.type == "mps":
        pipe.enable_attention_slicing()
    return pipe


def preprocess(img: Image.Image, lineart: LineartDetector, max_size: int) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    scale = max_size / max(w, h)
    new_w = int(w * scale) // 8 * 8
    new_h = int(h * scale) // 8 * 8
    img = img.resize((new_w, new_h), Image.LANCZOS)
    lineart_img = lineart(img)
    return img, lineart_img


def make_output_name(
    source_stem: str, flow: str, denoise: float, cn_strength: float, seed: int
) -> str:
    d = int(denoise * 100)
    cn = int(cn_strength * 10)
    return f"{source_stem}_{flow}_d{d:03d}_cn{cn:02d}_s{seed}.png"


def main():
    parser = argparse.ArgumentParser(description="Photo -> Stylized image via Diffusers + ControlNet")
    parser.add_argument("--image", required=True, help="Path to source photo")
    parser.add_argument("--flow", nargs="+", choices=list(FLOWS.keys()), default=list(FLOWS.keys()))
    parser.add_argument("--denoise", nargs="+", type=float, default=[0.6])
    parser.add_argument("--controlnet-strength", nargs="+", type=float, default=[0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=9)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--output-dir", default="./generated-images")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    models_dir = pathlib.Path(args.models_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = detect_device()
    print(f"Device: {device} | dtype: {dtype}")

    # Load lineart detector
    print("Loading LineartDetector ...")
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")

    # Load source image and preprocess
    source_path = pathlib.Path(args.image)
    source_img = Image.open(source_path).convert("RGB")
    img, lineart_img = preprocess(source_img, lineart, args.max_size)
    print(f"Preprocessed {source_path.name} -> {img.size}")

    # Figure out which checkpoints we need
    needed_checkpoints = set(FLOWS[f]["checkpoint"] for f in args.flow)

    # Load pipelines (one per checkpoint)
    pipelines: dict[str, StableDiffusionControlNetImg2ImgPipeline] = {}
    for ckpt_key in needed_checkpoints:
        ckpt_path = str(models_dir / CHECKPOINT_FILES[ckpt_key])
        print(f"Loading pipeline for {ckpt_key} ...")
        pipelines[ckpt_key] = load_pipeline(ckpt_path, device, dtype)
    print("All pipelines loaded.")

    # Generate
    combos = list(itertools.product(args.flow, args.denoise, args.controlnet_strength, args.seeds))
    total = len(combos)
    source_stem = source_path.stem
    print(f"Generating {total} images (steps={args.steps}, cfg={args.cfg}, max_size={args.max_size}) ...")

    for i, (flow, denoise, cn_strength, seed) in enumerate(combos, 1):
        out_name = make_output_name(source_stem, flow, denoise, cn_strength, seed)
        out_path = output_dir / out_name
        flow_cfg = FLOWS[flow]
        pipe = pipelines[flow_cfg["checkpoint"]]

        print(f"[{i}/{total}] {out_name} ...", end=" ", flush=True)
        t0 = time.time()

        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            prompt=flow_cfg["positive"],
            negative_prompt=flow_cfg["negative"],
            image=img,
            control_image=lineart_img,
            strength=denoise,
            controlnet_conditioning_scale=cn_strength,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        ).images[0]

        result.save(out_path)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

    print(f"\nAll {total} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
