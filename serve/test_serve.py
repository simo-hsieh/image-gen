#!/usr/bin/env python3
"""Lightweight test for Ray Serve topology — no GPU, no model downloads.

Validates the deployment graph wiring, request routing, and serialization
by stubbing out the heavy model calls.

Run: python test_serve.py
"""

import base64
import io

import requests
import torch
from PIL import Image
from ray import serve

from serve_diffusion import (
    FLOWS,
    DiffusionNode,
    Postprocessor,
    Preprocessor,
    Router,
    pil_to_bytes,
)


# -- Stub out heavy model loading ------------------------------------------

class FakeLineartDetector:
    """Returns a grayscale version instead of real lineart."""
    def __call__(self, img, **kwargs):
        return img.convert("L").convert("RGB")

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


class FakePipeline:
    """Returns a red-tinted version of the input instead of diffusion."""
    class FakeScheduler:
        config = {}
        @classmethod
        def from_config(cls, *a, **kw):
            return cls()

    scheduler = FakeScheduler()

    def __init__(self, **kw):
        pass

    def __call__(self, *, image, **kw):
        # Tint red so we can verify the image was processed
        r, g, b = image.split()
        result = Image.merge("RGB", (r, Image.new("L", image.size, 0), Image.new("L", image.size, 0)))
        return type("Out", (), {"images": [result]})()

    def to(self, device):
        return self

    def enable_attention_slicing(self):
        pass

    @classmethod
    def from_single_file(cls, *a, **kw):
        return cls()


# -- Monkey-patch before binding -------------------------------------------

import serve_diffusion
import controlnet_aux
import diffusers

# Patch the modules that serve_diffusion imported from
serve_diffusion.LineartDetector = FakeLineartDetector
serve_diffusion.StableDiffusionControlNetImg2ImgPipeline = FakePipeline
serve_diffusion.ControlNetModel = type("FakeCN", (), {"from_pretrained": classmethod(lambda cls, *a, **kw: None)})
serve_diffusion.DPMSolverMultistepScheduler = FakePipeline.FakeScheduler

# Force CPU
serve_diffusion.torch = torch


# -- Build and deploy ------------------------------------------------------

def make_test_image(w=64, h=64) -> bytes:
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    preprocessor = Preprocessor.bind()
    toonyou = DiffusionNode.bind(
        checkpoint_path="fake.safetensors",
        controlnet_id="fake/controlnet",
    )
    ghibli = DiffusionNode.bind(
        checkpoint_path="fake.ckpt",
        controlnet_id="fake/controlnet",
    )
    postprocessor = Postprocessor.bind()

    app = Router.bind(
        preprocessor=preprocessor,
        diffusion_nodes={"toonyou_beta6": toonyou, "ghibli_diffusion": ghibli},
        postprocessor=postprocessor,
    )

    handle = serve.run(app, host="127.0.0.1", port=8123)

    base_url = "http://127.0.0.1:8123"
    test_image_bytes = make_test_image()
    image_b64 = base64.b64encode(test_image_bytes).decode()

    # Test health
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200
    health = r.json()
    assert set(health["flows"]) == set(FLOWS.keys())
    print(f"[PASS] /health -> {health}")

    # Test each flow
    for flow in FLOWS:
        r = requests.post(f"{base_url}/generate", json={
            "image_b64": image_b64,
            "flow": flow,
            "denoise": 0.5,
            "controlnet_strength": 0.85,
            "seed": 42,
            "steps": 2,
            "cfg": 7.0,
            "max_size": 64,
        })
        assert r.status_code == 200, f"{flow} failed: {r.status_code} {r.text}"
        assert r.headers["content-type"] == "image/png"
        # Verify we got a valid image back
        img = Image.open(io.BytesIO(r.content))
        assert img.size[0] > 0
        print(f"[PASS] /generate flow={flow} -> {img.size[0]}x{img.size[1]} PNG")

    # Test invalid flow
    r = requests.post(f"{base_url}/generate", json={
        "image_b64": image_b64,
        "flow": "nonexistent",
    })
    assert r.status_code == 400
    print(f"[PASS] invalid flow -> 400")

    # Test bad base64
    r = requests.post(f"{base_url}/generate", json={
        "image_b64": "not-valid-base64!!!",
        "flow": "cartoon",
    })
    assert r.status_code == 400
    print(f"[PASS] bad base64 -> 400")

    print("\nAll tests passed!")
    serve.shutdown()


if __name__ == "__main__":
    main()
