.DEFAULT_GOAL := help

COMFY_DIR ?= $(shell cd .. && pwd)/ComfyUI

.PHONY: help setup setup-comfy comfy-start generate generate-comfy serve-ray serve-stop test clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Environment:"
	@echo "    make setup           Install Python dependencies with uv"
	@echo "    make setup-comfy     Clone ComfyUI + download models"
	@echo ""
	@echo "  ComfyUI (local dev):"
	@echo "    make comfy-start     Start ComfyUI server on :8188"
	@echo "    make generate-comfy  Generate images via ComfyUI API"
	@echo ""
	@echo "  Diffusers (standalone/production):"
	@echo "    make generate        Generate images via Diffusers"
	@echo "    make generate-ray    Generate via Ray Serve"
	@echo ""
	@echo "  Services:"
	@echo "    make serve-ray       Start Ray Serve on :8000"
	@echo "    make serve-stop      Stop Ray Serve"
	@echo ""
	@echo "  Testing:"
	@echo "    make test            Run tests"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean           Remove .venv"

# --- Environment setup ---

setup:
	uv sync

setup-comfy:
	bash scripts/setup-comfy.sh $(COMFY_DIR)

download-models:
	bash scripts/download-models.sh $(COMFY_DIR)/models

# --- ComfyUI (local dev) ---

comfy-start:
	cd $(COMFY_DIR) && python main.py

generate-comfy:
	uv run python comfy/generate_versions.py $(ARGS)

# --- Diffusers (standalone/production) ---

generate:
	MODELS_DIR=$(COMFY_DIR)/models uv run python generate.py $(ARGS)

generate-ray:
	uv run python serve/generate_versions_ray.py $(ARGS)

# --- Services ---

serve-ray:
	cd serve && uv run serve run serve_diffusion:app

serve-stop:
	uv run serve shutdown

# --- Testing ---

test:
	cd serve && uv run python test_serve.py

# --- Cleanup ---

clean:
	rm -rf .venv
