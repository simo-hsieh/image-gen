#!/usr/bin/env python3
"""Ray Serve deployment for image stylization pipeline.

Topological architecture with independently scalable stages:
  Preprocessor (CPU) -> DiffusionNode (GPU) -> Postprocessor (CPU)
  orchestrated by a Router ingress.

Run with:
    serve run serve_diffusion:app
    python serve_diffusion.py
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import torch
from controlnet_aux import LineartDetector
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
from fastapi import FastAPI
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle

logger = logging.getLogger("ray.serve")

# ---------------------------------------------------------------------------
# Flow definitions
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Serialization helpers (PIL <-> bytes across Ray actors)
# ---------------------------------------------------------------------------


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded source image")
    flow: str = Field("cartoon", description="Style flow name")
    denoise: float = Field(0.5, ge=0.0, le=1.0, description="Denoise / strength")
    controlnet_strength: float = Field(0.85, ge=0.0, le=1.0)
    seed: int = Field(42)
    steps: int = Field(15, ge=1)
    cfg: float = Field(7.0, ge=1.0)
    max_size: int = Field(512, ge=64)


# ---------------------------------------------------------------------------
# Stage 1 - Preprocessor (CPU)
# ---------------------------------------------------------------------------


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class Preprocessor:
    def __init__(self) -> None:
        logger.info("Loading LineartDetector ...")
        self.lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
        logger.info("LineartDetector ready.")

    async def __call__(self, image_bytes: bytes, max_size: int) -> dict[str, bytes]:
        """Resize and extract lineart edges. Returns serialized PIL images."""
        img = bytes_to_pil(image_bytes)

        # Resize longest side to max_size, dimensions rounded to multiples of 8
        w, h = img.size
        scale = max_size / max(w, h)
        new_w = int(w * scale) // 8 * 8
        new_h = int(h * scale) // 8 * 8
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Extract lineart control image
        lineart_img = self.lineart(img)

        return {
            "image": pil_to_bytes(img),
            "lineart": pil_to_bytes(lineart_img),
        }


# ---------------------------------------------------------------------------
# Stage 2 - DiffusionNode (GPU)
# ---------------------------------------------------------------------------


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class DiffusionNode:
    def __init__(self, checkpoint_path: str, controlnet_id: str) -> None:
        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float32
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        logger.info(
            "DiffusionNode loading checkpoint=%s controlnet=%s device=%s",
            checkpoint_path,
            controlnet_id,
            self.device,
        )

        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=self.dtype
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            checkpoint_path,
            controlnet=controlnet,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        if self.device.type == "mps":
            self.pipe.enable_attention_slicing()

        logger.info("DiffusionNode ready on %s", self.device)

    async def __call__(
        self,
        image_bytes: bytes,
        control_image_bytes: bytes,
        positive_prompt: str,
        negative_prompt: str,
        strength: float,
        controlnet_conditioning_scale: float,
        steps: int,
        cfg: float,
        seed: int,
    ) -> bytes:
        image = bytes_to_pil(image_bytes)
        control_image = bytes_to_pil(control_image_bytes)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=control_image,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

        return pil_to_bytes(result)


# ---------------------------------------------------------------------------
# Stage 3 - Postprocessor (CPU)
# ---------------------------------------------------------------------------


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class Postprocessor:
    async def __call__(self, image_bytes: bytes, output_format: str = "PNG") -> bytes:
        """Re-encode image to the desired output format."""
        img = bytes_to_pil(image_bytes)
        buf = io.BytesIO()
        img.save(buf, format=output_format)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Stage 4 - Router / Ingress (FastAPI)
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(title="Image Stylization API")


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi_app)
class Router:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        diffusion_nodes: dict[str, DeploymentHandle],
        postprocessor: DeploymentHandle,
    ) -> None:
        self.preprocessor = preprocessor
        self.diffusion_nodes = diffusion_nodes
        self.postprocessor = postprocessor

    @fastapi_app.get("/health")
    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "flows": list(FLOWS.keys()),
            "checkpoints": list(self.diffusion_nodes.keys()),
        }

    @fastapi_app.post("/generate")
    async def generate(self, req: GenerateRequest) -> Response:
        # Validate flow
        if req.flow not in FLOWS:
            return Response(
                content=f"Unknown flow '{req.flow}'. Available: {list(FLOWS.keys())}",
                status_code=400,
                media_type="text/plain",
            )

        flow_cfg = FLOWS[req.flow]
        checkpoint_key = flow_cfg["checkpoint"]

        if checkpoint_key not in self.diffusion_nodes:
            return Response(
                content=(
                    f"No DiffusionNode loaded for checkpoint '{checkpoint_key}'. "
                    f"Available: {list(self.diffusion_nodes.keys())}"
                ),
                status_code=500,
                media_type="text/plain",
            )

        # Decode input image
        try:
            raw_bytes = base64.b64decode(req.image_b64)
        except Exception:
            return Response(
                content="Invalid base64 image data.",
                status_code=400,
                media_type="text/plain",
            )

        # Stage 1 - Preprocess
        preprocess_result = await self.preprocessor.remote(raw_bytes, req.max_size)
        image_bytes = preprocess_result["image"]
        lineart_bytes = preprocess_result["lineart"]

        # Stage 2 - Diffusion
        node_handle = self.diffusion_nodes[checkpoint_key]
        result_bytes = await node_handle.remote(
            image_bytes,
            lineart_bytes,
            flow_cfg["positive"],
            flow_cfg["negative"],
            req.denoise,
            req.controlnet_strength,
            req.steps,
            req.cfg,
            req.seed,
        )

        # Stage 3 - Postprocess
        output_bytes = await self.postprocessor.remote(result_bytes, "PNG")

        return Response(content=output_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# Build deployment graph
# ---------------------------------------------------------------------------

preprocessor = Preprocessor.bind()
toonyou_node = DiffusionNode.bind(
    checkpoint_path="ComfyUI/models/checkpoints/toonyou_beta6.safetensors",
    controlnet_id="lllyasviel/control_v11p_sd15_lineart",
)
ghibli_node = DiffusionNode.bind(
    checkpoint_path="ComfyUI/models/checkpoints/ghibli_diffusion.ckpt",
    controlnet_id="lllyasviel/control_v11p_sd15_lineart",
)
postprocessor = Postprocessor.bind()

app = Router.bind(
    preprocessor=preprocessor,
    diffusion_nodes={"toonyou_beta6": toonyou_node, "ghibli_diffusion": ghibli_node},
    postprocessor=postprocessor,
)

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    serve.run(app, host="0.0.0.0", port=8000)
