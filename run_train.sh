#!/bin/bash
#SBATCH --job-name=P2Echo-new
#SBATCH --account=def-ilkerh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/moeinh78/P2Echo-new/logs/output-%j.txt
#SBATCH --error=/scratch/moeinh78/P2Echo-new/logs/error-%j.txt

set -euo pipefail

# ============================================================================
# P2Echo-new Training Script
# ============================================================================

cd /scratch/moeinh78/P2Echo-new
mkdir -p logs

# Activate environment
source ~/envs/sam3/bin/activate

nvidia-smi

# Fix matplotlib/fontconfig cache directory issues
export XDG_CONFIG_HOME=/scratch/moeinh78/P2Echo-new/.config
export XDG_CACHE_HOME=/scratch/moeinh78/P2Echo-new/.cache
export MPLCONFIGDIR=${XDG_CONFIG_HOME}/matplotlib
export FONTCONFIG_CACHE=${XDG_CACHE_HOME}/fontconfig
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_CACHE"

# Triton cache (torch.compile, flash-attn, etc.)
export TRITON_CACHE_DIR=/scratch/moeinh78/P2Echo-new/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# HuggingFace/cache and data locations
export HF_HOME=/project/def-ilkerh/moeinh78/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export SAM3_DATA_ROOT=/project/def-ilkerh/moeinh78/data
export SAM3_SPLITS_JSON=${SAM3_DATA_ROOT}/data_splits.json

# Force fully offline mode - uses cached files only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# PYTHONPATH for imports

# Control CPU thread usage (prevents libgomp thread failures)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Training Configuration
# ============================================================================

RUN_NAME=${RUN_NAME:-"p2echo_pvt_dgdecoder"}
TEXT_MODEL=${TEXT_MODEL:-"Qwen/Qwen3-Embedding-0.6B"}

echo "=============================================="
echo "[INFO] Starting P2Echo-new training..."
echo "  run_name: ${RUN_NAME}"
echo "  text_model: ${TEXT_MODEL}"
echo "=============================================="

# Run training
python src/train.py \
    --splits_json "${SAM3_SPLITS_JSON}" \
    --data_root "${SAM3_DATA_ROOT}" \
    --output_dir "./outputs" \
    --run_name "${RUN_NAME}" \
    --image_size 256 \
    --batch_size 8 \
    --num_workers 4 \
    --pretrained_encoder \
    --pretrained_dir "./pretrained_pth" \
    --text_model "${TEXT_MODEL}" \
    --text_cache_dir "${HF_HOME}/hub" \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --aug_clip 3.0 \
    --permute_prompts \
    --use_amp \
    --bf16 \
    --save_interval 10 \
    --val_interval 1 \
    --seed 42 \
    "$@"
