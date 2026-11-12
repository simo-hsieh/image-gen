#!/usr/bin/env bash
# Setup ComfyUI as an external service alongside this project.
# Usage: bash scripts/setup-comfy.sh [install-dir]
#
# Default install dir: ../ComfyUI (sibling to this repo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="${1:-$PROJECT_DIR/../ComfyUI}"

echo "==> ComfyUI install dir: $INSTALL_DIR"

# --- Clone or update ComfyUI ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "==> ComfyUI already cloned, pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "==> Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$INSTALL_DIR"
fi

# --- Install ComfyUI dependencies ---
echo "==> Installing ComfyUI dependencies..."
pip install -r "$INSTALL_DIR/requirements.txt"

# --- Download models ---
bash "$SCRIPT_DIR/download-models.sh" "$INSTALL_DIR/models"

# --- Symlink custom nodes (if any) ---
CUSTOM_NODES_SRC="$PROJECT_DIR/custom_nodes"
CUSTOM_NODES_DST="$INSTALL_DIR/custom_nodes"
if [ -d "$CUSTOM_NODES_SRC" ]; then
    echo "==> Symlinking custom nodes..."
    for node_dir in "$CUSTOM_NODES_SRC"/*/; do
        node_name="$(basename "$node_dir")"
        target="$CUSTOM_NODES_DST/$node_name"
        if [ -L "$target" ] || [ -d "$target" ]; then
            echo "    Already linked: $node_name"
        else
            ln -s "$node_dir" "$target"
            echo "    Linked: $node_name"
        fi
    done
fi

echo ""
echo "==> Done! Start ComfyUI with:"
echo "    cd $INSTALL_DIR && python main.py"
