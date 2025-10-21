#!/usr/bin/env bash
set -e

# Detect conda installation
_conda_sh="$(find "$HOME" /opt -type f -path '*/etc/profile.d/conda.sh' ! -path '*/pkgs/*' 2>/dev/null | head -n 1)"
CONDA_ROOT="$(echo "$_conda_sh" | grep -Eo '.*/(miniforge3|miniforge|mambaforge|miniconda3|miniconda|anaconda3|anaconda)' | head -n 1)"
[ -n "$CONDA_ROOT" ] && source "$CONDA_ROOT/etc/profile.d/conda.sh"

# --- Use conda ---
conda create -n vllm python=3.12 -y
conda activate vllm
pip install uv
uv pip install vllm --torch-backend=auto
uv pip install flashinfer-python flashinfer-cubin
