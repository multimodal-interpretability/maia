#!/usr/bin/env bash
set -euo pipefail

# Defaults
MODEL=""
GPUS=1
PORT=11434
HOST="0.0.0.0"
GPU_UTIL="0.90"
MAX_NUM_SEQS=1

usage() {
  cat <<EOF
Usage: $0 --model <mistral|gemma|hf_model_id> [--gpus N]

  --model   mistral | gemma | any other Hugging Face model id (generic run)
  --gpus    Number of GPUs to use (tensor parallel size). Default: 1

Examples:
  $0 --model mistral --gpus 2
  $0 --model gemma
  $0 --model meta-llama/Llama-3.1-8B-Instruct --gpus 1
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value"; usage; exit 1; }
      MODEL="$2"; shift 2;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "Error: --gpus requires a value"; usage; exit 1; }
      GPUS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Error: --model is required."
  usage
  exit 1
fi

# Build command based on model
declare -a CMD
case "${MODEL}" in
  mistral|Mistral)
    MODEL_ID="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    CMD=(
      vllm serve "${MODEL_ID}"
      --tokenizer-mode mistral
      --config-format mistral
      --load-format mistral
      --tool-call-parser mistral
      --dtype auto
      --tensor-parallel-size "${GPUS}"
      --gpu-memory-utilization "${GPU_UTIL}"
      --host "${HOST}"
      --port "${PORT}"
      --enable-chunked-prefill
      --enable-prefix-caching
      --max_num_seqs "${MAX_NUM_SEQS}"
      --served-model-name mistralai/Mistral-Small-3.2-24B
    )
    ;;
  gemma|Gemma)
    MODEL_ID="google/gemma-3-27b-it"
    CMD=(
      vllm serve "${MODEL_ID}"
      --dtype auto
      --tensor-parallel-size "${GPUS}"
      --gpu-memory-utilization "${GPU_UTIL}"
      --host "${HOST}"
      --port "${PORT}"
      --enable-chunked-prefill
      --enable-prefix-caching
      --max_num_seqs "${MAX_NUM_SEQS}"
    )
    ;;
  *)
    # Generic/default run with sensible defaults
    MODEL_ID="${MODEL}"
    CMD=(
      vllm serve "${MODEL_ID}"
      --dtype auto
      --tensor-parallel-size "${GPUS}"
      --gpu-memory-utilization "${GPU_UTIL}"
      --host "${HOST}"
      --port "${PORT}"
      --enable-chunked-prefill
      --enable-prefix-caching
      --max_num_seqs "${MAX_NUM_SEQS}"
    )
    ;;
esac

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
