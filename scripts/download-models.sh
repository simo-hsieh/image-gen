#!/usr/bin/env bash
# Download models from HuggingFace into a models directory.
# Usage: bash scripts/download-models.sh [models-dir]
#
# Default models dir: ../ComfyUI/models (sibling to this repo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${1:-$PROJECT_DIR/../ComfyUI/models}"

CHECKPOINTS_DIR="$MODELS_DIR/checkpoints"
CONTROLNET_DIR="$MODELS_DIR/controlnet"
mkdir -p "$CHECKPOINTS_DIR" "$CONTROLNET_DIR"

echo "==> Downloading models to $MODELS_DIR ..."

echo "  checkpoints/toonyou_beta6.safetensors"
hf download frankjoshua/toonyou_beta6 toonyou_beta6.safetensors \
    --local-dir "$CHECKPOINTS_DIR"

echo "  controlnet/control_v11p_sd15_lineart.pth"
hf download lllyasviel/ControlNet-v1-1 control_v11p_sd15_lineart.pth \
    --local-dir "$CONTROLNET_DIR"

echo "==> Done. Models saved to $MODELS_DIR"
