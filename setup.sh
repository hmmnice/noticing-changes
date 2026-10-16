#!/usr/bin/env bash
# One-time setup: Python venv with PyTorch (CUDA), Depth Anything 3 and CLIP.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q torch torchvision opencv-python scipy transformers huggingface_hub safetensors \
    einops addict omegaconf plyfile imageio trimesh e3nn pillow_heif evo pycolmap "moviepy==1.0.3"
[ -d third_party/da3 ] || { git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3 third_party/da3 && git -C third_party/da3 checkout -q 3d835ec; }
.venv/bin/pip install -q --no-deps -e third_party/da3
echo "ready: .venv/bin/python changes.py before.mp4 after.mp4 --out results/my_room"
