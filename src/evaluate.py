"""
Evaluation script for P2Echo-new.

Evaluates the best checkpoint on the external split (EchoNet-Dynamic and HMCQU).
Produces per-dataset metrics (Dice, IoU, HD95, ASSD) and qualitative figures.

Usage:
    python src/evaluate.py \
        --checkpoint ./outputs/p2echo_v2_small_transformer_boundary_loss/checkpoints/best.pth \
        --splits_json /project/def-ilkerh/moeinh78/data/data_splits.json \
        --data_root /project/def-ilkerh/moeinh78/data \
        --text_model Qwen/Qwen3-Embedding-0.6B \
        --output_dir ./outputs/eval_external
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data import load_splits_json, make_loaders
from data.dataset import MultiTaskEchoDataset, normalize_image
from metrics import MEDPY_AVAILABLE, MetricAggregator
from prompts import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    PromptSpec,
    build_prompt_text_from_view,
    build_background_prompt_text,
    default_prompt_specs,
    present_label_set,
    view_label_set,
)
from networks.p2echo.net import P2Echo
from networks.p2echo.text_encoder import FrozenTextBackbone


# =============================================================================
# Utilities
# =============================================================================

def resolve_hf_local_snapshot(model_id_or_path: str) -> str:
    """If running in offline mode, resolve HF model id to local snapshot path."""
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


def infer_decoder_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, bool]:
    """
    Infer decoder configuration from parameter names in a checkpoint state_dict.

    Returns:
        (decoder_type, ita_dual_injection)
    """
    keys = state_dict.keys()
    has_dyita = any(".dyita." in k for k in keys)
    has_dyita_dec4 = any(k.startswith("decoder.dec4.dyita.") for k in keys)
    has_inject_convs = any(k.startswith("decoder.inject_convs.") for k in keys)

    if has_dyita_dec4:
        return "ita_nocfa", has_inject_convs
    if has_dyita:
        return "ita", has_inject_convs
    return "cenet", False


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log(line: str, log_path: Optional[Path] = None) -> None:
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(line + "\n")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Text embedding (same as train.py)
# =============================================================================

def build_text_embeddings(
    *,
    text_backbone: FrozenTextBackbone,
    planes: Sequence[str],
    prompt_specs: Sequence[PromptSpec],
    device: torch.device,
) -> torch.Tensor:
    """Build text embeddings for a batch. Returns [B, N, D]."""
    prompt_texts: List[List[str]] = []
    for plane in planes:
        per: List[str] = []
        for spec in prompt_specs:
            per.append(
                build_prompt_text_from_view(
                    plane=str(plane),
                    spec=spec,
                    include_present_absent_context=True,
                )
            )
        prompt_texts.append(per)

    flat = [p for per in prompt_texts for p in per]
    emb_flat = text_backbone.embed_prompts(flat, device=device)
    b = len(planes)
    n = len(prompt_specs)
    return emb_flat.view(b, n, -1)


# =============================================================================
# Prediction utilities (same as train.py)
# =============================================================================

def denorm_img(img: torch.Tensor, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Convert a normalized tensor into a displayable grayscale image in [0, 1]."""
    x = img.detach().cpu().float()
    if x.ndim == 3:
        x = x.mean(dim=0)
    x_np = x.numpy()
    if not np.isfinite(x_np).all():
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(np.min(x_np)), float(np.max(x_np))
    if vmin >= 0.0 and vmax <= 1.0:
        out = x_np
    elif vmin >= -1.5 and vmax <= 1.5:
        out = x_np * 0.5 + 0.5
    else:
        lo = float(np.percentile(x_np, float(p_low)))
        hi = float(np.percentile(x_np, float(p_high)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
            out = np.zeros_like(x_np, dtype=np.float32)
        else:
            out = (x_np - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def preds_to_multiclass_mask(
    *,
    probs: np.ndarray,
    prompt_specs: Sequence[PromptSpec],
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert binary-per-prompt probabilities into a single multi-class mask."""
    h, w = probs.shape[-2:]
    class_ids = [1, 2, 3, 4, 5]
    class_prob = np.full((len(class_ids), h, w), -np.inf, dtype=np.float32)
    for pi, spec in enumerate(prompt_specs):
        lids = tuple(int(x) for x in spec.label_ids if int(x) != 0)
        if len(lids) != 1:
            continue
        lid = lids[0]
        if lid in class_ids:
            class_prob[class_ids.index(lid)] = np.maximum(
                class_prob[class_ids.index(lid)], probs[pi].astype(np.float32)
            )
    max_prob = np.max(class_prob, axis=0)
    arg = np.argmax(class_prob, axis=0)
    out = np.zeros((h, w), dtype=np.uint8)
    out[max_prob >= float(threshold)] = np.array(class_ids, dtype=np.uint8)[
        arg[max_prob >= float(threshold)]
    ]
    return out


def mask_probs_for_labels(
    *,
    probs: np.ndarray,
    prompt_specs: Sequence[PromptSpec],
    valid_label_ids: Sequence[int],
) -> np.ndarray:
    """Zero out probabilities for labels not present/visible for a given sample."""
    keep = set(int(x) for x in valid_label_ids)
    masked = probs.copy()
    for pi, spec in enumerate(prompt_specs):
        lids = [int(x) for x in spec.label_ids if int(x) != 0]
        if len(lids) != 1:
            continue
        if lids[0] not in keep:
            masked[pi] = 0.0
    return masked


def suppress_predictions_for_gt_absent_classes(
    *,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Set predictions to BG for classes that are absent in this sample's GT.

    This mirrors U-Bench-style metric postprocessing and is useful for
    apples-to-apples comparison across codebases.
    """
    out = pred_mask.copy()
    present_ids = set(int(x) for x in np.unique(gt_mask))
    if class_ids is None:
        class_ids = [1, 2, 3, 4, 5]
    for cid in class_ids:
        cid_int = int(cid)
        if cid_int != 0 and cid_int not in present_ids:
            out[out == cid_int] = 0
    return out


# =============================================================================
# Qualitative visualization
# =============================================================================

OVERLAY_COLORS = [
    (0.0, 0.0, 0.0, 0.0),    # 0 BG transparent
    (0.0, 1.0, 1.0, 0.45),   # 1 LV  cyan
    (1.0, 0.75, 0.8, 0.45),  # 2 MYO pink
    (0.56, 0.93, 0.56, 0.45),# 3 LA  green
    (1.0, 0.65, 0.0, 0.45),  # 4 RV  orange
    (0.58, 0.0, 0.83, 0.45), # 5 RA  purple
]

LABEL_NAMES = ["BG", "LV", "MYO", "LA", "RV", "RA"]


def save_qualitative_grid(
    *,
    out_path: Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str,
    valid_class_ids: Sequence[int],
) -> None:
    """Save a 3-panel qualitative figure: Image | GT overlay | Pred overlay."""
    _mkdir(out_path.parent)

    cmap = ListedColormap([c[:3] for c in OVERLAY_COLORS])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax in axs:
        ax.axis("off")

    axs[0].set_title("Image", fontsize=13)
    axs[0].imshow(image, cmap="gray", vmin=0, vmax=1)

    axs[1].set_title("Ground Truth", fontsize=13)
    axs[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    gt_ma = np.ma.masked_where(gt_mask == 0, gt_mask)
    axs[1].imshow(gt_ma, cmap=cmap, vmin=0, vmax=5)

    axs[2].set_title("Prediction", fontsize=13)
    axs[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    pr_ma = np.ma.masked_where(pred_mask == 0, pred_mask)
    axs[2].imshow(pr_ma, cmap=cmap, vmin=0, vmax=5)

    # Build legend only for valid classes
    legend_patches = []
    for cid in valid_class_ids:
        legend_patches.append(
            Patch(facecolor=OVERLAY_COLORS[cid][:3], edgecolor="none", label=LABEL_NAMES[cid])
        )
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               frameon=True, fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(
    *,
    model: P2Echo,
    text_backbone: FrozenTextBackbone,
    loader,
    prompt_specs: Sequence[PromptSpec],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    logits_mode: str = "sigmoid",
    suppress_predictions_for_classes_absent_in_ground_truth: bool = True,
    threshold: float = 0.5,
    max_qual_per_dataset: int = 10,
    output_dir: Path,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run evaluation on a dataloader and save metrics + qualitative figures."""
    if not MEDPY_AVAILABLE:
        raise RuntimeError("medpy is required for evaluation. Install: pip install medpy")

    model.eval()
    agg = MetricAggregator()
    qual_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    n_batches = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        planes = batch["plane"]
        datasets = batch["dataset"]
        stems = batch.get("stem", [""] * img.shape[0])

        b = img.shape[0]

        # Build text embeddings
        text_emb = build_text_embeddings(
            text_backbone=text_backbone,
            planes=planes,
            prompt_specs=prompt_specs,
            device=device,
        )

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(img, text_emb)
        else:
            outputs = model(img, text_emb)

        # Get main output
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
        else:
            logits = outputs

        if logits_mode == "softmax":
            probs = F.softmax(logits, dim=1).detach().float().cpu().numpy()
        else:
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        gt_np = gt.detach().cpu().numpy().astype(np.uint8)

        for i in range(b):
            ds = str(datasets[i])
            view = str(planes[i])
            present = set(present_label_set(ds))
            visible = set(view_label_set(view))
            valid_labels = sorted(
                LABEL_TO_ID[l] for l in (present & visible)
                if l in LABEL_TO_ID and l != "BG"
            )

            masked_probs = mask_probs_for_labels(
                probs=probs[i],
                prompt_specs=prompt_specs,
                valid_label_ids=valid_labels,
            )

            if logits_mode == "softmax":
                pred_mc = np.argmax(masked_probs, axis=0).astype(np.uint8)
            else:
                pred_mc = preds_to_multiclass_mask(
                    probs=masked_probs,
                    prompt_specs=prompt_specs,
                    threshold=threshold,
                )
            if suppress_predictions_for_classes_absent_in_ground_truth:
                pred_mc = suppress_predictions_for_gt_absent_classes(
                    pred_mask=pred_mc,
                    gt_mask=gt_np[i],
                    class_ids=valid_labels,
                )

            if valid_labels:
                agg.update(
                    pred_mc,
                    gt_np[i],
                    view=view,
                    dataset=ds,
                    class_ids=valid_labels,
                )

            # Collect qualitative examples
            if len(qual_examples[ds]) < max_qual_per_dataset:
                qual_examples[ds].append({
                    "image": img[i].cpu(),
                    "gt": gt_np[i],
                    "pred": pred_mc,
                    "plane": view,
                    "stem": str(stems[i]) if stems else "",
                    "valid_class_ids": valid_labels,
                })

        n_batches += 1
        if n_batches % 20 == 0:
            _log(f"[{_now()}]   Processed {n_batches} batches...", log_path)

    # =========================================================================
    # Save qualitative figures
    # =========================================================================
    _log(f"\n[{_now()}] Saving qualitative figures...", log_path)
    qual_dir = output_dir / "qualitative"
    for ds_name, examples in sorted(qual_examples.items()):
        for ex in examples:
            img_np = denorm_img(ex["image"])
            out_path = qual_dir / ds_name / f"{ex['stem']}_{ex['plane']}.png"
            save_qualitative_grid(
                out_path=out_path,
                image=img_np,
                gt_mask=ex["gt"],
                pred_mask=ex["pred"],
                title=f"{ds_name}  |  {ex['plane']}  |  {ex['stem']}",
                valid_class_ids=ex["valid_class_ids"],
            )
        _log(f"  Saved {len(examples)} figures for {ds_name}", log_path)

    # =========================================================================
    # Compute and log metrics
    # =========================================================================
    metrics = agg.to_dict()

    _log(f"\n{'='*80}", log_path)
    _log(f"EVALUATION RESULTS  ({metrics['n_samples']} samples)", log_path)
    _log(f"{'='*80}", log_path)

    # Overall
    _log(_format_summary_table(summary=metrics["overall"], title="OVERALL"), log_path)

    # Mean metrics
    mm = metrics.get("mean_metrics", {})
    _log(
        f"\nMean Dice: {mm.get('mean_dice', float('nan')):.4f}  |  "
        f"Mean IoU: {mm.get('mean_iou', float('nan')):.4f}  |  "
        f"Mean HD95: {mm.get('mean_hd95', float('nan')):.2f}  |  "
        f"Mean ASSD: {mm.get('mean_assd', float('nan')):.2f}",
        log_path,
    )

    # Per-dataset breakdown
    per_dataset = metrics.get("per_dataset", {})
    for ds_name in sorted(per_dataset.keys()):
        _log(f"\n{'-'*60}", log_path)
        _log(
            _format_summary_table(
                summary=per_dataset[ds_name],
                title=f"Dataset: {ds_name}",
            ),
            log_path,
        )

    # Per-view breakdown
    per_view = metrics.get("per_view", {})
    if per_view:
        for view_name in sorted(per_view.keys()):
            _log(f"\n{'-'*60}", log_path)
            _log(
                _format_summary_table(
                    summary=per_view[view_name],
                    title=f"View: {view_name}",
                ),
                log_path,
            )

    return metrics


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
        n = m.get("dice_n", 0)
        lines.append(
            f"{cls:>8} | {dice:>10.4f} | {iou:>10.4f} | {hd95:>10.2f} | {assd:>10.2f}   (n={n})"
        )
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate P2Echo-new on external data")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to data_splits.json")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root directory for data paths")
    parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Text encoder model name or path")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_external",
                        help="Directory to save evaluation results")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_pth")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="auto",
        choices=["auto", "cenet", "ita", "ita_nocfa", "dyITA_NoCFA"],
        help=(
            "Decoder type to instantiate. 'auto' infers from checkpoint keys. "
            "Use 'ita_nocfa'/'dyITA_NoCFA' for DyITA bottleneck without CFAModule."
        ),
    )
    parser.add_argument("--ita_dual_injection", action="store_true",
                        help="For decoder_type in {ita, ita_nocfa}: enable post-hoc text injection after ITABlock.")
    parser.add_argument("--logits_mode", type=str, default="sigmoid", choices=["sigmoid", "softmax"],
                        help="How to decode logits: sigmoid-threshold (binary prompts) or softmax-argmax (multiclass).")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_qual", type=int, default=10,
                        help="Max qualitative figures per dataset")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary masks")
    parser.add_argument(
        "--suppress_predictions_for_classes_absent_in_ground_truth",
        action="store_true",
        help=(
            "For metric computation, set predicted pixels to background for classes "
            "that are absent in each sample's ground-truth mask."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    _mkdir(output_dir)
    log_path = output_dir / "eval_log.txt"

    _log(f"[{_now()}] P2Echo-new External Evaluation", log_path)
    _log(f"[{_now()}] Checkpoint: {args.checkpoint}", log_path)
    _log(f"[{_now()}] Args: {args}", log_path)

    # =========================================================================
    # Load data (external split only)
    # =========================================================================
    _log(f"\n[{_now()}] Loading external data split...", log_path)
    _, _, _, external_df = load_splits_json(args.splits_json, data_root=args.data_root)

    if external_df is None or len(external_df) == 0:
        _log("[ERROR] No external data found in the splits JSON!", log_path)
        return

    # Show dataset distribution
    ds_counts = external_df["dataset"].value_counts()
    _log(f"[{_now()}] External split: {len(external_df)} total samples", log_path)
    for ds_name, count in ds_counts.items():
        _log(f"  {ds_name}: {count} samples", log_path)

    external_loader = torch.utils.data.DataLoader(
        MultiTaskEchoDataset(
            external_df,
            resize=(args.image_size, args.image_size),
            to_rgb=True,
            image_transform=normalize_image,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # =========================================================================
    # Prompt specs
    # =========================================================================
    prompt_specs = default_prompt_specs()
    _log(f"[{_now()}] Using {len(prompt_specs)} prompt specs", log_path)
    for i, spec in enumerate(prompt_specs):
        label_names = [ID_TO_LABEL.get(lid, f"id{lid}") for lid in spec.label_ids]
        _log(f"  [{i}] labels={label_names}", log_path)

    # =========================================================================
    # Text backbone
    # =========================================================================
    resolved_text_model = resolve_hf_local_snapshot(args.text_model)
    _log(f"[{_now()}] Loading text backbone: {resolved_text_model}", log_path)
    text_backbone = FrozenTextBackbone(model_name=resolved_text_model)
    text_backbone.to(device)

    # =========================================================================
    # Model
    # =========================================================================
    _log(f"[{_now()}] Loading checkpoint: {args.checkpoint}", log_path)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    inferred_decoder_type, inferred_ita_dual = infer_decoder_config_from_state_dict(state_dict)
    decoder_type = inferred_decoder_type if args.decoder_type == "auto" else args.decoder_type
    if decoder_type == "dyITA_NoCFA":
        decoder_type = "ita_nocfa"
    ita_dual_injection = bool(args.ita_dual_injection) or (
        args.decoder_type == "auto" and inferred_ita_dual
    )

    _log(f"[{_now()}] Building P2Echo model...", log_path)
    _log(
        f"[{_now()}] Decoder config: decoder_type={decoder_type}, "
        f"ita_dual_injection={ita_dual_injection}",
        log_path,
    )
    _log(f"[{_now()}] Logits decode mode: {args.logits_mode}", log_path)
    _log(
        f"[{_now()}] Suppress GT-absent predicted classes: "
        f"{args.suppress_predictions_for_classes_absent_in_ground_truth}",
        log_path,
    )
    num_classes = len(prompt_specs)
    model = P2Echo(
        input_channels=3,
        img_size=(args.image_size, args.image_size),
        encoder_name="pvt_v2_b2",
        pretrained_encoder=False,  # loading from checkpoint
        pretrained_dir=args.pretrained_dir,
        text_embedding_dim=text_backbone.embedding_dim,
        num_classes=num_classes,
        deep_supervision=True,
        decoder_type=decoder_type,
        ita_dual_injection=ita_dual_injection,
    )

    # Load checkpoint weights
    model.load_state_dict(state_dict)
    model = model.to(device)

    best_dice = ckpt.get("best_dice", float("nan"))
    ckpt_epoch = ckpt.get("epoch", -1)
    _log(f"[{_now()}] Checkpoint from epoch {ckpt_epoch}, best_dice={best_dice:.4f}", log_path)

    total_params = sum(p.numel() for p in model.parameters())
    _log(f"[{_now()}] Total model params: {total_params:,}", log_path)

    # =========================================================================
    # Run evaluation
    # =========================================================================
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    _log(f"\n[{_now()}] Starting evaluation on external split...", log_path)
    metrics = evaluate(
        model=model,
        text_backbone=text_backbone,
        loader=external_loader,
        prompt_specs=prompt_specs,
        device=device,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        logits_mode=args.logits_mode,
        suppress_predictions_for_classes_absent_in_ground_truth=(
            args.suppress_predictions_for_classes_absent_in_ground_truth
        ),
        threshold=args.threshold,
        max_qual_per_dataset=args.max_qual,
        output_dir=output_dir,
        log_path=log_path,
    )

    _log(f"\n[{_now()}] Evaluation complete. Results saved to: {output_dir}", log_path)
    _log(f"[{_now()}] Log: {log_path}", log_path)
    _log(f"[{_now()}] Qualitative figures: {output_dir / 'qualitative'}", log_path)


if __name__ == "__main__":
    main()
