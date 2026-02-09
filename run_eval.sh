#!/bin/bash
#SBATCH --job-name=P2Echo-eval
#SBATCH --account=def-ilkerh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/moeinh78/P2Echo-new/logs/eval-output-%j.txt
#SBATCH --error=/scratch/moeinh78/P2Echo-new/logs/eval-error-%j.txt

# ============================================================================
# P2Echo-new Evaluation Script — External Split
# ============================================================================

cd /scratch/moeinh78/P2Echo-new
mkdir -p logs

# Activate environment
source ~/envs/sam3/bin/activate || true

nvidia-smi || true

# Fix matplotlib/fontconfig cache directory issues
export XDG_CONFIG_HOME=/scratch/moeinh78/P2Echo-new/.config
export XDG_CACHE_HOME=/scratch/moeinh78/P2Echo-new/.cache
export MPLCONFIGDIR=${XDG_CONFIG_HOME}/matplotlib
export FONTCONFIG_CACHE=${XDG_CACHE_HOME}/fontconfig
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_CACHE"

# Triton cache
export TRITON_CACHE_DIR=/scratch/moeinh78/P2Echo-new/.triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# HuggingFace/cache and data locations
export HF_HOME=/project/def-ilkerh/moeinh78/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export SAM3_DATA_ROOT=/project/def-ilkerh/moeinh78/data
export SAM3_SPLITS_JSON=${SAM3_DATA_ROOT}/data_splits.json

# Force fully offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# CPU thread control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Evaluation Configuration
# ============================================================================

CHECKPOINT="./outputs/p2echo_v2_small_transformer_boundary_loss/checkpoints/best.pth"
OUTPUT_DIR="./outputs/eval_external"
TEXT_MODEL="Qwen/Qwen3-Embedding-0.6B"

echo "=============================================="
echo "[INFO] Starting P2Echo-new evaluation..."
echo "  checkpoint: ${CHECKPOINT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "=============================================="

python src/evaluate.py \
    --checkpoint "${CHECKPOINT}" \
    --splits_json "${SAM3_SPLITS_JSON}" \
    --data_root "${SAM3_DATA_ROOT}" \
    --text_model "${TEXT_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_size 256 \
    --batch_size 16 \
    --num_workers 8 \
    --pretrained_dir "./pretrained_pth" \
    --use_amp \
    --bf16 \
    --max_qual 10 \
    --threshold 0.5 \
    --seed 42 \
    "$@"
