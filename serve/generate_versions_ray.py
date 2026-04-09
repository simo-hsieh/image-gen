#!/usr/bin/env python3
"""Convert a personal photo into cartoon/anime/Ghibli style images via Ray Serve."""

import argparse
import base64
import itertools
import pathlib

import requests

SERVER = "127.0.0.1:8000"

FLOWS = {
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
        "positive": "ghibli style, studio ghibli anime screenshot, miyazaki, soft watercolor, black hair, glasses, handsome young asian man protagonist, soft facial features, gentle face, clear smooth skin, kind bright eyes, warm gentle smile, anime hero, ikemen, bishounen, black jacket, masterpiece, best quality, detailed face, beautiful warm lighting",
        "negative": "realistic, 3d render, deformed face, bad anatomy, yellow hair, blonde hair, low quality, blurry, ugly, aged, wrinkles, dark circles, tired looking, harsh features, rough skin, angular face, villain",
        "checkpoint": "ghibli_diffusion",
    },
}


def make_output_name(
    source_stem: str, flow: str, denoise: float, cn_strength: float, seed: int
) -> str:
    """Build systematic output filename per spec."""
    d = int(denoise * 100)
    cn = int(cn_strength * 10)
    return f"{source_stem}_{flow}_d{d:03d}_cn{cn:02d}_s{seed}.png"


def generate_one(
    server: str,
    image_path: str,
    flow: str,
    denoise: float,
    controlnet_strength: float,
    seed: int,
    steps: int,
    cfg: float,
    max_size: int,
) -> bytes:
    """POST to the Ray Serve /generate endpoint and return PNG bytes."""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_b64": image_base64,
        "flow": flow,
        "denoise": denoise,
        "controlnet_strength": controlnet_strength,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "max_size": max_size,
    }

    resp = requests.post(f"http://{server}/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.content


def main():
    parser = argparse.ArgumentParser(
        description="Photo -> Cartoon style generator via Ray Serve"
    )
    parser.add_argument("--image", required=True, help="Path to source photo")
    parser.add_argument(
        "--flow",
        nargs="+",
        choices=list(FLOWS.keys()),
        default=list(FLOWS.keys()),
        help="Style flows to generate",
    )
    parser.add_argument(
        "--denoise",
        nargs="+",
        type=float,
        default=[0.45, 0.5, 0.55],
        help="Denoise strengths",
    )
    parser.add_argument(
        "--controlnet-strength",
        nargs="+",
        type=float,
        default=[0.85],
        help="ControlNet strengths",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42], help="Seeds for generation"
    )
    parser.add_argument(
        "--output-dir", default="./versions", help="Output directory"
    )
    parser.add_argument(
        "--steps", type=int, default=15, help="KSampler steps (default: 15)"
    )
    parser.add_argument(
        "--cfg", type=float, default=7, help="CFG scale (default: 7)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Max image dimension in px (default: 512)",
    )
    parser.add_argument(
        "--server", default=SERVER, help="Ray Serve server address"
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = pathlib.Path(args.image).stem

    combos = list(
        itertools.product(
            args.flow, args.denoise, args.controlnet_strength, args.seeds
        )
    )
    total = len(combos)
    print(f"Generating {total} variations (steps={args.steps}, max_size={args.max_size})...")

    for i, (flow, denoise, cn_strength, seed) in enumerate(combos, 1):
        out_name = make_output_name(source_stem, flow, denoise, cn_strength, seed)
        out_path = output_dir / out_name
        print(f"[{i}/{total}] {out_name} ...", end=" ", flush=True)

        png_bytes = generate_one(
            server=args.server,
            image_path=args.image,
            flow=flow,
            denoise=denoise,
            controlnet_strength=cn_strength,
            seed=seed,
            steps=args.steps,
            cfg=args.cfg,
            max_size=args.max_size,
        )
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print("done")

    print(f"All {total} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
