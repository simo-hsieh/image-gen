#!/usr/bin/env python3
"""Convert a personal photo into cartoon/anime/Ghibli style images using ComfyUI."""

import argparse
import itertools
import json
import os
import pathlib
import urllib.parse
import urllib.request
import uuid

import websocket  # websocket-client

SERVER = "127.0.0.1:8188"

CHECKPOINT = "toonyou_beta6.safetensors"
CONTROLNET = "control_v11p_sd15_lineart.pth"
LORA = "ghibli_style_offset.safetensors"

FLOWS = {
    "cartoon": {
        "positive": "cartoon style, soft colors, gentle cel shading, yellow hair, blonde hair, soft rounded chin",
        "negative": "sharp chin, angular jaw, dark hair, realistic",
    },
    "pixar": {
        "positive": "pixar style, 3d render, soft lighting, yellow hair, soft rounded chin",
        "negative": "sharp chin, dark hair, flat",
    },
    "anime": {
        "positive": "anime style, pastel colors, studio ghibli, yellow hair, soft rounded chin",
        "negative": "sharp chin, dark hair, western cartoon",
    },
    "ghibli": {
        "positive": "ghibli style, miyazaki, soft watercolor, yellow hair, soft rounded chin",
        "negative": "sharp chin, dark hair, 3d render",
    },
}


def upload_image(server: str, filepath: str) -> str:
    """Upload an image to ComfyUI and return the server-side filename."""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        image_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"http://{server}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["name"]


def build_workflow(
    flow: str,
    uploaded_filename: str,
    denoise: float,
    cn_strength: float,
    seed: int,
) -> dict:
    """Build the ComfyUI workflow dict for img2img + ControlNet."""
    use_lora = flow == "ghibli"
    prompts = FLOWS[flow]

    # Where model and clip come from
    model_source = ["11", 0] if use_lora else ["1", 0]
    clip_source = ["11", 1] if use_lora else ["1", 1]

    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_filename},
        },
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompts["positive"],
                "clip": clip_source,
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompts["negative"],
                "clip": clip_source,
            },
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": CONTROLNET},
        },
        "7": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "control_net": ["6", 0],
                "image": ["2", 0],
                "strength": cn_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": model_source,
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["3", 0],
                "seed": seed,
                "steps": 25,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["1", 2],
            },
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": "cartoon",
            },
        },
    }

    if use_lora:
        workflow["11"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": LORA,
                "strength_model": 0.85,
                "strength_clip": 0.85,
            },
        }

    return workflow


def queue_prompt(server: str, workflow: dict, client_id: str) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def wait_for_completion(server: str, client_id: str, prompt_id: str) -> None:
    """Block until ComfyUI finishes executing the given prompt."""
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server}/ws?clientId={client_id}")
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "executing":
                    exec_data = data.get("data", {})
                    if (
                        exec_data.get("prompt_id") == prompt_id
                        and exec_data.get("node") is None
                    ):
                        break
    finally:
        ws.close()


def fetch_and_save(server: str, prompt_id: str, output_path: str) -> None:
    """Download generated images from ComfyUI history and save to disk."""
    url = f"http://{server}/history/{prompt_id}"
    with urllib.request.urlopen(url) as resp:
        history = json.loads(resp.read())

    outputs = history[prompt_id]["outputs"]
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            params = urllib.parse.urlencode(
                {"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")}
            )
            img_url = f"http://{server}/view?{params}"
            with urllib.request.urlopen(img_url) as img_resp:
                with open(output_path, "wb") as f:
                    f.write(img_resp.read())


def make_output_name(
    source_stem: str, flow: str, denoise: float, cn_strength: float, seed: int
) -> str:
    """Build systematic output filename per spec."""
    d = int(denoise * 100)
    cn = int(cn_strength * 10)
    return f"{source_stem}_{flow}_d{d:03d}_cn{cn:02d}_s{seed}.png"


def main():
    parser = argparse.ArgumentParser(description="Photo → Cartoon style generator via ComfyUI")
    parser.add_argument("--image", required=True, help="Path to source photo")
    parser.add_argument("--flow", nargs="+", choices=list(FLOWS.keys()), default=list(FLOWS.keys()), help="Style flows to generate")
    parser.add_argument("--denoise", nargs="+", type=float, default=[0.45, 0.5, 0.55], help="Denoise strengths")
    parser.add_argument("--controlnet-strength", nargs="+", type=float, default=[0.85], help="ControlNet strengths")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for generation")
    parser.add_argument("--output-dir", default="./versions", help="Output directory")
    parser.add_argument("--server", default=SERVER, help="ComfyUI server address")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = pathlib.Path(args.image).stem
    client_id = str(uuid.uuid4())

    combos = list(itertools.product(args.flow, args.denoise, args.controlnet_strength, args.seeds))
    total = len(combos)
    print(f"Generating {total} variations...")

    for i, (flow, denoise, cn_strength, seed) in enumerate(combos, 1):
        out_name = make_output_name(source_stem, flow, denoise, cn_strength, seed)
        out_path = output_dir / out_name
        print(f"[{i}/{total}] {out_name} ...", end=" ", flush=True)

        uploaded = upload_image(args.server, args.image)
        workflow = build_workflow(flow, uploaded, denoise, cn_strength, seed)
        prompt_id = queue_prompt(args.server, workflow, client_id)
        wait_for_completion(args.server, client_id, prompt_id)
        fetch_and_save(args.server, prompt_id, str(out_path))
        print("done")

    print(f"All {total} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
