"""
Training script for P2Echo-new.

PVT-v2-B2 encoder + DGDecoder with text conditioning.
Binary per-prompt paradigm matching original VoxTell.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# matplotlib must be configured before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data import load_splits_json, make_loaders
from data.augment import build_nnunet_2d_train_transforms, apply_batchgenerators_transforms
from losses import DeepSupervisionDiceBCELoss
from metrics import MEDPY_AVAILABLE, MetricAggregator
from prompts import (
    LABEL_TO_ID,
    PromptSpec,
    build_background_prompt_text,
    build_prompt_text_from_view,
    default_prompt_specs,
    make_binary_targets,
    present_label_set,
)
from networks.p2echo.net import P2Echo
from networks.p2echo.text_encoder import FrozenTextBackbone


# =============================================================================
# Utilities
# =============================================================================

def resolve_hf_local_snapshot(model_id_or_path: str) -> str:
    """
    If running in offline mode, resolve HF model id to local snapshot path.
    """
    p = Path(model_id_or_path)
    if p.exists():
        return str(p)

    if "/" not in model_id_or_path:
        return model_id_or_path

    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        return model_id_or_path

    hub = Path(hf_home) / "hub"
    repo_dir = hub / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        return model_id_or_path

    snaps = sorted(repo_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    if not snaps:
        return model_id_or_path
    return str(snaps[0])


def _as_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log(line: str, log_path: Path) -> None:
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Metrics
# =============================================================================

def _format_summary_table(
    *,
    summary: Dict[str, Dict[str, float]],
    title: str,
) -> str:
    class_order = ["LV", "MYO", "LA", "RV", "RA"]
    header = f"{'Class':>8} | {'Dice':>10} | {'IoU':>10} | {'HD95':>10} | {'ASSD':>10}"
    lines = [title, header, "-" * len(header)]

    mean_acc = {"dice": [], "iou": [], "hd95": [], "assd": []}
    for cls in class_order:
        m = summary.get(cls)
        if not m:
            continue
        dice = m.get("dice_mean", float("nan"))
        iou = m.get("iou_mean", float("nan"))
        hd95 = m.get("hd95_mean", float("nan"))
        assd = m.get("assd_mean", float("nan"))
        lines.append(f"{cls:>8} | {dice:>10.4f} | {iou:>10.4f} | {hd95:>10.2f} | {assd:>10.2f}")
        if np.isfinite(dice):
            mean_acc["dice"].append(dice)
        if np.isfinite(iou):
            mean_acc["iou"].append(iou)
        if np.isfinite(hd95):
            mean_acc["hd95"].append(hd95)
        if np.isfinite(assd):
            mean_acc["assd"].append(assd)

    if any(mean_acc.values()):
        mean_dice = float(np.mean(mean_acc["dice"])) if mean_acc["dice"] else float("nan")
        mean_iou = float(np.mean(mean_acc["iou"])) if mean_acc["iou"] else float("nan")
        mean_hd95 = float(np.mean(mean_acc["hd95"])) if mean_acc["hd95"] else float("nan")
        mean_assd = float(np.mean(mean_acc["assd"])) if mean_acc["assd"] else float("nan")
        lines.append("-" * len(header))
        lines.append(
            f"{'Mean':>8} | {mean_dice:>10.4f} | {mean_iou:>10.4f} | {mean_hd95:>10.2f} | {mean_assd:>10.2f}"
        )

    return "\n".join(lines)


def _format_grouped_tables(
    *,
    grouped: Dict[str, Dict[str, Dict[str, float]]],
    title: str,
) -> str:
    if not grouped:
        return f"{title}\n(no data)"
    lines = ["=" * 80, title, "=" * 80]
    for key in sorted(grouped.keys()):
        lines.append("")
        lines.append(_format_summary_table(summary=grouped[key], title=f"{key}:"))
    return "\n".join(lines)


# =============================================================================
# Qualitative Visualization
# =============================================================================

def denorm_img(img: torch.Tensor) -> np.ndarray:
    """
    img: [3,H,W] normalized by (x-0.5)/0.5 -> roughly [-1,1]
    returns grayscale [H,W] in [0,1]
    """
    x = img.detach().cpu().float()
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0.0, 1.0)
    x = x.mean(dim=0)  # grayscale
    return x.numpy()


def preds_to_multiclass_mask(
    *,
    probs: np.ndarray,  # [N,H,W] in [0,1]
    prompt_specs: Sequence[PromptSpec],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert binary-per-prompt probabilities into a single multi-class mask for visualization.
    Uses only single-label prompts and selects argmax across classes; pixels with max<threshold -> BG.
    """
    h, w = probs.shape[-2:]
    class_ids = [1, 2, 3, 4, 5]
    class_prob = np.full((len(class_ids), h, w), -np.inf, dtype=np.float32)

    for pi, spec in enumerate(prompt_specs):
        lids = tuple(int(x) for x in spec.label_ids if int(x) != 0)
        if len(lids) != 1:
            continue
        lid = lids[0]
        if lid in class_ids:
            class_prob[class_ids.index(lid)] = np.maximum(class_prob[class_ids.index(lid)], probs[pi].astype(np.float32))

    max_prob = np.max(class_prob, axis=0)
    arg = np.argmax(class_prob, axis=0)  # 0..4
    out = np.zeros((h, w), dtype=np.uint8)
    out[max_prob >= float(threshold)] = np.array(class_ids, dtype=np.uint8)[arg[max_prob >= float(threshold)]]
    return out


def save_qualitative_grid(
    *,
    out_path: Path,
    image: np.ndarray,  # [H,W]
    gt_mask: np.ndarray,  # [H,W] uint8 0..5
    pred_mask: np.ndarray,  # [H,W] uint8 0..5
    title: str,
) -> None:
    _mkdir(out_path.parent)

    colors = [
        (0.0, 0.0, 0.0, 0.0),  # BG transparent
        (0.0, 1.0, 1.0, 0.35),  # LV
        (1.0, 0.75, 0.8, 0.35),  # MYO
        (0.56, 0.93, 0.56, 0.35),  # LA
        (1.0, 0.65, 0.0, 0.35),  # RV
        (0.58, 0.0, 0.83, 0.35),  # RA
    ]

    cmap = ListedColormap([c[:3] for c in colors])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax in axs:
        ax.axis("off")

    axs[0].set_title("Image")
    axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)

    axs[1].set_title("GT overlay")
    axs[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    gt_ma = np.ma.masked_where(gt_mask == 0, gt_mask)
    axs[1].imshow(gt_ma, cmap=cmap, vmin=0, vmax=5)

    axs[2].set_title("Pred overlay")
    axs[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    pr_ma = np.ma.masked_where(pred_mask == 0, pred_mask)
    axs[2].imshow(pr_ma, cmap=cmap, vmin=0, vmax=5)

    legend = [
        Patch(facecolor=colors[1][:3], edgecolor="none", label="LV"),
        Patch(facecolor=colors[2][:3], edgecolor="none", label="MYO"),
        Patch(facecolor=colors[3][:3], edgecolor="none", label="LA"),
        Patch(facecolor=colors[4][:3], edgecolor="none", label="RV"),
        Patch(facecolor=colors[5][:3], edgecolor="none", label="RA"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=True)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_qualitative_per_dataset(
    *,
    out_dir: Path,
    examples: Dict[str, List[Dict[str, Any]]],
    prompt_specs: Sequence[PromptSpec],
    threshold: float = 0.5,
) -> None:
    for dataset, items in sorted(examples.items()):
        for ex in items:
            img = ex["image"]  # torch [3,H,W]
            gt = ex["gt"]  # torch [H,W]
            probs = ex["probs"]  # torch [N,H,W]
            plane = str(ex.get("plane", ""))
            stem = str(ex.get("stem", ""))

            img_np = denorm_img(img)
            gt_np = gt.detach().cpu().numpy().astype(np.uint8)
            probs_np = probs.detach().cpu().float().numpy()
            pred_np = preds_to_multiclass_mask(probs=probs_np, prompt_specs=prompt_specs, threshold=threshold)

            out_path = out_dir / dataset / f"{stem}_{plane}.png"
            title = f"{dataset} | {plane} | {stem}"
            save_qualitative_grid(out_path=out_path, image=img_np, gt_mask=gt_np, pred_mask=pred_np, title=title)


# =============================================================================
# Text Embedding and Prompt Utilities
# =============================================================================

def build_text_embeddings(
    *,
    text_backbone: FrozenTextBackbone,
    planes: Sequence[str],
    gt_mask: torch.Tensor,
    prompt_specs: Sequence[PromptSpec],
    device: torch.device,
    prompt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build text embeddings for a batch. Returns [B, N, D].
    """
    prompt_texts = _build_batch_prompt_texts_from_gt(
        planes=planes,
        gt_mask=gt_mask,
        prompt_specs=prompt_specs,
        prompt_mask=prompt_mask,
    )
    flat = [p for per in prompt_texts for p in per]
    emb_flat = text_backbone.embed_prompts(flat, device=device)  # [B*N, D]
    b = len(planes)
    n = len(prompt_specs)
    return emb_flat.view(b, n, -1)


def _build_batch_prompt_texts_from_gt(
    *,
    planes: Sequence[str],
    gt_mask: torch.Tensor,
    prompt_specs: Sequence[PromptSpec],
    prompt_mask: torch.Tensor | None,
) -> List[List[str]]:
    out: List[List[str]] = []
    for bi, p in enumerate(planes):
        per: List[str] = []
        for pi, spec in enumerate(prompt_specs):
            if prompt_mask is not None and float(prompt_mask[bi, pi].item()) < 0.5:
                per.append(build_background_prompt_text(plane=str(p)))
                continue
            per.append(
                build_prompt_text_from_view(
                    plane=str(p),
                    spec=spec,
                    include_present_absent_context=True,
                )
            )
        out.append(per)
    return out


def build_valid_prompt_mask(
    *,
    datasets: Sequence[str],
    prompt_specs: Sequence[PromptSpec],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a [B,N] mask indicating which prompts are annotated for each dataset.

    We treat "not in this dataset's label map" as *unlabeled* and ignore its loss.
    """
    b = len(datasets)
    n = len(prompt_specs)
    mask = torch.zeros((b, n), device=device, dtype=torch.float32)
    for bi, ds in enumerate(datasets):
        present = set(present_label_set(str(ds)))
        present_ids = {int(LABEL_TO_ID[p]) for p in present if p in LABEL_TO_ID}
        for pi, spec in enumerate(prompt_specs):
            ids = [int(x) for x in spec.label_ids if int(x) != 0]
            if ids and all(lid in present_ids for lid in ids):
                mask[bi, pi] = 1.0
    return mask


# =============================================================================
# Prompt Permutation
# =============================================================================

def build_random_prompt_permutation(*, batch_size: int, num_prompts: int, device: torch.device) -> torch.Tensor:
    return torch.stack([torch.randperm(num_prompts, device=device) for _ in range(batch_size)], dim=0)


def apply_prompt_permutation(text_emb: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    idx = perm[:, :, None].expand(-1, -1, text_emb.shape[-1])
    return text_emb.gather(1, idx)


def invert_prompt_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(perm.shape[1], device=perm.device).expand_as(perm))
    return inv


def unpermute_outputs(outputs: List[torch.Tensor] | torch.Tensor, perm: torch.Tensor) -> List[torch.Tensor] | torch.Tensor:
    inv = invert_prompt_permutation(perm)
    out_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    unpermuted: List[torch.Tensor] = []
    for out in out_list:
        inv_idx = inv[:, :, None, None].expand(-1, -1, out.shape[2], out.shape[3])
        unpermuted.append(out.gather(1, inv_idx))
    if isinstance(outputs, (list, tuple)):
        return unpermuted
    return unpermuted[0]


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    *,
    model: P2Echo,
    text_backbone: FrozenTextBackbone,
    loader,
    prompt_specs: Sequence[PromptSpec],
    loss_fn: DeepSupervisionDiceBCELoss,
    optim: torch.optim.Optimizer,
    device: torch.device,
    train_aug,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float = 0.0,
    aug_clip: float = 0.0,
    permute_prompts: bool = True,
    log_path: Path | None = None,
    epoch: int = 0,
) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for batch_idx, batch in enumerate(loader):
        img = batch["image"]  # [B,3,H,W]
        gt = batch["mask"]  # [B,H,W]
        planes = batch["plane"]
        datasets = batch["dataset"]

        # Augmentation (batchgenerators) on CPU
        if train_aug is not None:
            img_np = img.detach().cpu().numpy().astype(np.float32)
            seg_np = gt.detach().cpu().numpy()[:, None].astype(np.int16)
            out = apply_batchgenerators_transforms(trf=train_aug, data=img_np, seg=seg_np)
            if not np.isfinite(out["data"]).all():
                print(f"[{_now()}] WARNING: non-finite values after augmentation at batch={batch_idx}; sanitizing.", flush=True)
                out["data"] = np.nan_to_num(out["data"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if aug_clip > 0.0:
                out["data"] = np.clip(out["data"], -aug_clip, aug_clip).astype(np.float32, copy=False)
            img = torch.from_numpy(out["data"])
            gt = torch.from_numpy(out["seg"][:, 0]).long()

        img = img.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        b = img.shape[0]
        n = len(prompt_specs)

        # Build valid_mask for this batch
        valid_mask = build_valid_prompt_mask(datasets=datasets, prompt_specs=prompt_specs, device=device)

        # Build binary targets from multi-class GT
        binary_targets = make_binary_targets(gt_mask=gt, prompt_specs=prompt_specs)  # [B,N,H,W]

        # Build text embeddings
        text_emb = build_text_embeddings(
            text_backbone=text_backbone,
            planes=planes,
            gt_mask=gt,
            prompt_specs=prompt_specs,
            device=device,
        )

        # Prompt permutation (if enabled)
        if permute_prompts:
            perm = build_random_prompt_permutation(batch_size=b, num_prompts=n, device=device)
            text_emb = apply_prompt_permutation(text_emb, perm)
            # Permute targets and valid_mask too
            binary_targets = binary_targets.gather(1, perm[:, :, None, None].expand(-1, -1, binary_targets.shape[2], binary_targets.shape[3]))
            valid_mask = valid_mask.gather(1, perm)

        optim.zero_grad()

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(img, text_emb)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                loss = loss_fn(outputs, binary_targets, valid_mask=valid_mask)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            outputs = model(img, text_emb)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            loss = loss_fn(outputs, binary_targets, valid_mask=valid_mask)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        loss_sum += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            msg = f"[{_now()}] Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}"
            if log_path:
                _log(msg, log_path)
            else:
                print(msg, flush=True)

    return loss_sum / max(n_batches, 1)


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    *,
    model: P2Echo,
    text_backbone: FrozenTextBackbone,
    loader,
    prompt_specs: Sequence[PromptSpec],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    threshold: float = 0.5,
    max_qual_per_dataset: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    if not MEDPY_AVAILABLE:
        raise RuntimeError("medpy is required for validation metrics. Install medpy or disable medpy-style metrics.")

    model.eval()

    agg = MetricAggregator()
    qual_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        planes = batch["plane"]
        datasets = batch["dataset"]
        stems = batch.get("stem", [""] * img.shape[0])

        b = img.shape[0]
        n = len(prompt_specs)

        # Build text embeddings
        text_emb = build_text_embeddings(
            text_backbone=text_backbone,
            planes=planes,
            gt_mask=gt,
            prompt_specs=prompt_specs,
            device=device,
        )

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(img, text_emb)
        else:
            outputs = model(img, text_emb)

        # Get main output (highest resolution)
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]  # [B,N,H,W]
        else:
            logits = outputs

        probs = torch.sigmoid(logits)  # [B,N,H,W]
        probs_np = probs.detach().float().cpu().numpy()
        gt_np = gt.detach().cpu().numpy().astype(np.uint8)

        # Per-case medpy metrics (multiclass)
        for i in range(b):
            pred_mc = preds_to_multiclass_mask(
                probs=probs_np[i],
                prompt_specs=prompt_specs,
                threshold=threshold,
            )
            agg.update(
                pred_mc,
                gt_np[i],
                view=str(planes[i]),
                dataset=str(datasets[i]),
            )

        # Collect qualitative examples
        for i in range(b):
            ds = str(datasets[i])
            if len(qual_examples[ds]) < max_qual_per_dataset:
                qual_examples[ds].append({
                    "image": img[i].cpu(),
                    "gt": gt[i].cpu(),
                    "probs": probs[i].cpu(),
                    "plane": str(planes[i]),
                    "stem": str(stems[i]) if stems else "",
                })

    metrics = agg.to_dict()
    return metrics, qual_examples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train P2Echo-new")
    
    # Data
    parser.add_argument("--splits_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_pth")
    parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--text_cache_dir", type=str, default="/project/def-ilkerh/moeinh78/.cache/huggingface/hub/")
    
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aug_clip", type=float, default=3.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--permute_prompts", action="store_true")
    parser.add_argument("--disable_aug", action="store_true")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=1)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"p2echo_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    _mkdir(output_dir)
    _mkdir(output_dir / "checkpoints")
    _mkdir(output_dir / "qualitative")
    
    log_path = output_dir / "train.log"
    _log(f"[{_now()}] Starting training run: {run_name}", log_path)
    _log(f"[{_now()}] Args: {args}", log_path)
    
    # Data
    _log(f"[{_now()}] Loading data splits...", log_path)
    train_df, val_df, test_df, external_df = load_splits_json(args.splits_json, data_root=args.data_root)
    train_loader, val_loader, _, _ = make_loaders(
        train_df=train_df,
        val_df=val_df,
        resize=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _log(f"[{_now()}] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}", log_path)
    
    # Augmentation
    train_aug = None
    if not args.disable_aug:
        train_aug = build_nnunet_2d_train_transforms(patch_size=(args.image_size, args.image_size))
    
    # Prompt specs
    prompt_specs = default_prompt_specs()
    _log(f"[{_now()}] Using {len(prompt_specs)} prompt specs", log_path)
    
    # Text backbone - resolve to local snapshot if offline
    resolved_text_model = resolve_hf_local_snapshot(args.text_model)
    _log(f"[{_now()}] Loading text backbone: {resolved_text_model}", log_path)
    text_backbone = FrozenTextBackbone(
        model_name=resolved_text_model,
    )
    text_backbone.to(device)
    
    # Model
    _log(f"[{_now()}] Building P2Echo model...", log_path)
    model = P2Echo(
        input_channels=3,
        img_size=(args.image_size, args.image_size),
        encoder_name="pvt_v2_b2",
        pretrained_encoder=args.pretrained_encoder,
        pretrained_dir=args.pretrained_dir,
        text_embedding_dim=text_backbone.embedding_dim,
        deep_supervision=True,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"[{_now()}] Total params: {total_params:,}, Trainable: {trainable_params:,}", log_path)
    
    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler (poly)
    def poly_lr(epoch: int) -> float:
        return (1 - epoch / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, poly_lr)
    
    # Loss
    loss_fn = DeepSupervisionDiceBCELoss()
    
    # AMP
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume and Path(args.resume).exists():
        _log(f"[{_now()}] Resuming from {args.resume}", log_path)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        _log(f"\n[{_now()}] ===== Epoch {epoch}/{args.epochs} =====", log_path)
        _log(f"[{_now()}] LR: {optim.param_groups[0]['lr']:.6f}", log_path)
        
        train_loss = train_one_epoch(
            model=model,
            text_backbone=text_backbone,
            loader=train_loader,
            prompt_specs=prompt_specs,
            loss_fn=loss_fn,
            optim=optim,
            device=device,
            train_aug=train_aug,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            aug_clip=args.aug_clip,
            permute_prompts=args.permute_prompts,
            log_path=log_path,
            epoch=epoch,
        )
        _log(f"[{_now()}] Train loss: {train_loss:.4f}", log_path)
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            metrics, qual_examples = validate(
                model=model,
                text_backbone=text_backbone,
                loader=val_loader,
                prompt_specs=prompt_specs,
                device=device,
                use_amp=args.use_amp,
                amp_dtype=amp_dtype,
            )

            mean_metrics = metrics.get("mean_metrics", {})
            mean_dice = mean_metrics.get("mean_dice", float("nan"))
            mean_iou = mean_metrics.get("mean_iou", float("nan"))
            mean_hd95 = mean_metrics.get("mean_hd95", float("nan"))
            mean_assd = mean_metrics.get("mean_assd", float("nan"))
            _log(
                f"[{_now()}] Val Mean Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, "
                f"HD95: {mean_hd95:.2f}, ASSD: {mean_assd:.2f}",
                log_path,
            )

            overall_summary = metrics.get("overall", {})
            _log(_format_summary_table(summary=overall_summary, title="OVERALL METRICS"), log_path)

            per_view = metrics.get("per_view", {})
            _log(_format_grouped_tables(grouped=per_view, title="PER-VIEW BREAKDOWN"), log_path)

            per_dataset = metrics.get("per_dataset", {})
            _log(_format_grouped_tables(grouped=per_dataset, title="PER-DATASET BREAKDOWN"), log_path)
            
            # Save qualitative
            save_qualitative_per_dataset(
                out_dir=output_dir / "qualitative" / f"epoch_{epoch:04d}",
                examples=qual_examples,
                prompt_specs=prompt_specs,
            )
            
            # Check best
            is_best = np.isfinite(mean_dice) and mean_dice > best_dice
            if is_best:
                best_dice = mean_dice
                _log(f"[{_now()}] New best dice: {best_dice:.4f}", log_path)
            
            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "metrics": metrics,
            }
            torch.save(ckpt, output_dir / "checkpoints" / "latest.pth")
            if is_best:
                torch.save(ckpt, output_dir / "checkpoints" / "best.pth")
            if (epoch + 1) % args.save_interval == 0:
                torch.save(ckpt, output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth")
    
    _log(f"\n[{_now()}] Training complete. Best dice: {best_dice:.4f}", log_path)
    _log(f"[{_now()}] Results saved to: {output_dir}", log_path)


if __name__ == "__main__":
    main()
