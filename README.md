# P2Echo-new

Short guide to run training and understand the layout.

## Quick Start

1) Activate your environment (example):
```bash
source ~/envs/sam3/bin/activate
```

2) Run training:
```bash
sbatch run_train.sh
```

You can override options by appending flags in `run_train.sh` or passing them directly:
```bash
python src/train.py --splits_json /path/to/data_splits.json --data_root /path/to/data --epochs 200
```

## Repository Structure

- `run_train.sh`: SLURM training entrypoint (sets env, cache paths, and runs `src/train.py`).
- `src/train.py`: main training/validation loop.
- `src/networks/p2echo/`: model components.
  - `net.py`: P2Echo model (PVT-v2-B2 encoder + DGDecoder + text conditioning).
  - `decoders/`: DGDecoder and Mamba-based blocks.
  - `text_encoder.py`: frozen Qwen embedding backbone.
- `src/data/`: dataset loading and augmentation.
- `src/prompts.py`: prompt generation and label mappings.
- `src/losses.py`: loss functions.
- `src/metrics.py`: medpy-based evaluation utilities.
- `pretrained_pth/`: pretrained encoder weights.

## Outputs

Training artifacts are written to `./outputs/<run_name>/` with logs, checkpoints, and qualitative images.
