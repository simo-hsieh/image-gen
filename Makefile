.PHONY: env env-comfy env-serve serve serve-stop generate generate-comfy generate-ray test test-serve clean

# --- Environment setup ---

env:
	@echo "Usage:"
	@echo "  make env-comfy   Create ComfyUI conda environment"
	@echo "  make env-serve   Create Ray Serve conda environment"
	@echo "  make clean       Remove both environments"

env-comfy:
	conda env create -f environment-comfy.yml --yes

env-serve:
	conda env create -f environment-serve.yml --yes

# --- Ray Serve ---

serve:
	cd serve && conda run -n image-gen-serve serve run serve_diffusion:app

serve-stop:
	conda run -n image-gen-serve serve shutdown

# --- Image generation ---

generate:
	@echo "Usage:"
	@echo "  make generate-ray    Generate via Ray Serve (requires 'make serve' running)"
	@echo "  make generate-comfy  Generate via ComfyUI (requires ComfyUI running on :8188)"
	@echo ""
	@echo "Pass extra args with ARGS, e.g. make generate-ray ARGS=\"--flow ghibli\""

generate-ray:
	conda run -n image-gen-serve python serve/generate_versions_ray.py $(ARGS)

generate-comfy:
	conda run -n image-gen-comfy python comfy/generate_versions.py $(ARGS)

# --- Testing ---

test:
	@echo "Usage:"
	@echo "  make test-serve  Run Ray Serve topology tests (no GPU required)"

test-serve:
	cd serve && conda run -n image-gen-serve python test_serve.py

# --- Cleanup ---

clean:
	conda env remove -n image-gen-comfy --yes
	conda env remove -n image-gen-serve --yes
