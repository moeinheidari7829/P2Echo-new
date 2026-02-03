"""
CENet-style metrics for P2Echo-new.

Uses medpy library for per-case metric computation.
Provides per-view and per-dataset breakdown.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# medpy for CENet-style metrics
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]
try:
    from medpy.metric.binary import dc, jc, hd95, assd
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    warnings.warn(
        "medpy not installed. Install with: pip install medpy"
    )


# =============================================================================
# Label constants
# =============================================================================

LABEL_TO_ID: Dict[str, int] = {
    "BG": 0,
    "LV": 1,
    "MYO": 2,
    "LA": 3,
    "RV": 4,
    "RA": 5,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

# Foreground classes (exclude background)
FG_CLASSES = ["LV", "MYO", "LA", "RV", "RA"]
FG_CLASS_IDS = [LABEL_TO_ID[c] for c in FG_CLASSES]


# =============================================================================
# Per-case metric computation (CENet-style)
# =============================================================================

def compute_case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_id: int,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute metrics for a single case and single class.
    
    Uses medpy functions following CENet methodology:
    - Dice coefficient (dc)
    - Jaccard index / IoU (jc)
    - 95th percentile Hausdorff distance (hd95)
    - Average symmetric surface distance (assd)
    
    Args:
        pred: [H, W] predicted segmentation (integer labels)
        target: [H, W] ground truth segmentation (integer labels)
        class_id: Class ID to evaluate
        voxel_spacing: Pixel spacing for distance metrics
    
    Returns:
        Dict with 'dice', 'iou', 'hd95', 'assd' values
    """
    if not MEDPY_AVAILABLE:
        raise RuntimeError("medpy is required for CENet-style metrics")
    
    # Extract binary masks for this class
    pred_binary = (pred == class_id).astype(np.uint8)
    target_binary = (target == class_id).astype(np.uint8)
    
    # Handle empty cases
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    if target_sum == 0 and pred_sum == 0:
        # Both empty - perfect match
        return {
            "dice": 1.0,
            "iou": 1.0,
            "hd95": 0.0,
            "assd": 0.0,
        }
    elif target_sum == 0:
        # Target empty but prediction not - false positives
        return {
            "dice": 0.0,
            "iou": 0.0,
            "hd95": float("inf"),
            "assd": float("inf"),
        }
    elif pred_sum == 0:
        # Prediction empty but target not - false negatives
        return {
            "dice": 0.0,
            "iou": 0.0,
            "hd95": float("inf"),
            "assd": float("inf"),
        }
    
    # Compute metrics using medpy
    dice_val = dc(pred_binary, target_binary)
    iou_val = jc(pred_binary, target_binary)
    
    try:
        hd95_val = hd95(pred_binary, target_binary, voxelspacing=voxel_spacing)
    except Exception:
        hd95_val = float("inf")
    
    try:
        assd_val = assd(pred_binary, target_binary, voxelspacing=voxel_spacing)
    except Exception:
        assd_val = float("inf")
    
    return {
        "dice": float(dice_val),
        "iou": float(iou_val),
        "hd95": float(hd95_val),
        "assd": float(assd_val),
    }


def compute_multiclass_case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_ids: Optional[Sequence[int]] = None,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all classes in a single case.
    
    Args:
        pred: [H, W] predicted segmentation
        target: [H, W] ground truth segmentation
        class_ids: List of class IDs to evaluate (default: all foreground)
        voxel_spacing: Pixel spacing for distance metrics
    
    Returns:
        Dict mapping class_name -> metric_dict
    """
    if class_ids is None:
        class_ids = FG_CLASS_IDS
    
    results = {}
    for cid in class_ids:
        class_name = ID_TO_LABEL.get(cid, f"class_{cid}")
        results[class_name] = compute_case_metrics(
            pred, target, cid, voxel_spacing
        )
    
    return results


# =============================================================================
# Metric Aggregator
# =============================================================================

@dataclass
class MetricAggregator:
    """
    Aggregates metrics across samples with per-view and per-dataset breakdown.
    
    Following CENet methodology: computes per-case metrics, then macro-averages.
    """
    
    # Per-class metrics: class_name -> metric_name -> list of values
    per_class: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    # Per-view metrics: view -> class_name -> metric_name -> list of values
    per_view: Dict[str, Dict[str, Dict[str, List[float]]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Per-dataset metrics: dataset -> class_name -> metric_name -> list of values
    per_dataset: Dict[str, Dict[str, Dict[str, List[float]]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Sample count
    n_samples: int = 0
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.per_class = defaultdict(lambda: defaultdict(list))
        self.per_view = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.per_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.n_samples = 0
    
    def update(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        view: Optional[str] = None,
        dataset: Optional[str] = None,
        class_ids: Optional[Sequence[int]] = None,
        voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[str, Dict[str, float]]:
        """
        Update aggregator with a single sample.
        
        Args:
            pred: [H, W] predicted segmentation
            target: [H, W] ground truth segmentation
            view: View name (e.g., '4CH', '2CH')
            dataset: Dataset name (e.g., 'Camus', 'EchoNet-Dynamic')
            class_ids: Class IDs to evaluate
            voxel_spacing: Pixel spacing
        
        Returns:
            Per-class metrics for this sample
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        if class_ids is None:
            class_ids = FG_CLASS_IDS
        
        # Compute per-class metrics for this sample
        case_metrics = compute_multiclass_case_metrics(
            pred, target, class_ids, voxel_spacing
        )
        
        # Aggregate
        for class_name, metrics in case_metrics.items():
            for metric_name, value in metrics.items():
                # Skip infinite values in aggregation
                if not np.isfinite(value):
                    continue
                
                self.per_class[class_name][metric_name].append(value)
                
                if view is not None:
                    self.per_view[view][class_name][metric_name].append(value)
                
                if dataset is not None:
                    self.per_dataset[dataset][class_name][metric_name].append(value)
        
        self.n_samples += 1
        return case_metrics
    
    def update_batch(
        self,
        preds: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        views: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        class_ids: Optional[Sequence[int]] = None,
        voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Update aggregator with a batch of samples.
        
        Args:
            preds: [B, H, W] predicted segmentations
            targets: [B, H, W] ground truth segmentations
            views: List of view names (length B)
            datasets: List of dataset names (length B)
            class_ids: Class IDs to evaluate
            voxel_spacing: Pixel spacing
        
        Returns:
            List of per-sample, per-class metrics
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        batch_size = preds.shape[0]
        
        if views is None:
            views = [None] * batch_size
        if datasets is None:
            datasets = [None] * batch_size
        
        all_metrics = []
        for i in range(batch_size):
            metrics = self.update(
                preds[i], targets[i],
                view=views[i],
                dataset=datasets[i],
                class_ids=class_ids,
                voxel_spacing=voxel_spacing,
            )
            all_metrics.append(metrics)
        
        return all_metrics
    
    def _compute_summary(
        self,
        data: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean and std for each class/metric."""
        summary = {}
        for class_name, metrics in data.items():
            summary[class_name] = {}
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    summary[class_name][f"{metric_name}_mean"] = float(np.mean(values))
                    summary[class_name][f"{metric_name}_std"] = float(np.std(values))
                    summary[class_name][f"{metric_name}_n"] = len(values)
        return summary
    
    def get_overall_summary(self) -> Dict[str, Dict[str, float]]:
        """Get overall per-class metrics (macro-averaged across all samples)."""
        return self._compute_summary(self.per_class)
    
    def get_per_view_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get per-view, per-class metrics."""
        return {
            view: self._compute_summary(data)
            for view, data in self.per_view.items()
        }
    
    def get_per_dataset_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get per-dataset, per-class metrics."""
        return {
            dataset: self._compute_summary(data)
            for dataset, data in self.per_dataset.items()
        }
    
    def get_mean_metrics(self) -> Dict[str, float]:
        """
        Get overall mean metrics averaged across all foreground classes.
        
        Returns dict with: mean_dice, mean_iou, mean_hd95, mean_assd
        """
        summary = self.get_overall_summary()
        
        metrics = defaultdict(list)
        for class_name, class_metrics in summary.items():
            for metric_key, value in class_metrics.items():
                if metric_key.endswith("_mean"):
                    base_name = metric_key.replace("_mean", "")
                    if np.isfinite(value):
                        metrics[base_name].append(value)
        
        result = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                result[f"mean_{metric_name}"] = float(np.mean(values))
        
        return result
    
    def to_dict(self) -> Dict:
        """Export all metrics to a dictionary."""
        return {
            "n_samples": self.n_samples,
            "overall": self.get_overall_summary(),
            "mean_metrics": self.get_mean_metrics(),
            "per_view": self.get_per_view_summary(),
            "per_dataset": self.get_per_dataset_summary(),
        }
    
    def format_table(self, include_per_view: bool = True) -> str:
        """Format metrics as a readable table string."""
        lines = []
        
        # Overall metrics
        lines.append("=" * 80)
        lines.append("OVERALL METRICS (macro-averaged)")
        lines.append("=" * 80)
        
        summary = self.get_overall_summary()
        header = f"{'Class':>8} | {'Dice':>10} | {'IoU':>10} | {'HD95':>10} | {'ASSD':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        
        for class_name in FG_CLASSES:
            if class_name in summary:
                m = summary[class_name]
                dice = m.get("dice_mean", float("nan"))
                iou = m.get("iou_mean", float("nan"))
                hd95 = m.get("hd95_mean", float("nan"))
                assd_val = m.get("assd_mean", float("nan"))
                lines.append(
                    f"{class_name:>8} | {dice:>10.4f} | {iou:>10.4f} | {hd95:>10.2f} | {assd_val:>10.2f}"
                )
        
        # Mean metrics
        mean_metrics = self.get_mean_metrics()
        lines.append("-" * len(header))
        lines.append(
            f"{'Mean':>8} | {mean_metrics.get('mean_dice', float('nan')):>10.4f} | "
            f"{mean_metrics.get('mean_iou', float('nan')):>10.4f} | "
            f"{mean_metrics.get('mean_hd95', float('nan')):>10.2f} | "
            f"{mean_metrics.get('mean_assd', float('nan')):>10.2f}"
        )
        
        # Per-view breakdown
        if include_per_view:
            per_view = self.get_per_view_summary()
            if per_view:
                lines.append("")
                lines.append("=" * 80)
                lines.append("PER-VIEW BREAKDOWN")
                lines.append("=" * 80)
                
                for view in sorted(per_view.keys()):
                    view_summary = per_view[view]
                    lines.append(f"\n{view}:")
                    lines.append(header)
                    lines.append("-" * len(header))
                    
                    view_dice_values = []
                    for class_name in FG_CLASSES:
                        if class_name in view_summary:
                            m = view_summary[class_name]
                            dice = m.get("dice_mean", float("nan"))
                            iou = m.get("iou_mean", float("nan"))
                            hd95 = m.get("hd95_mean", float("nan"))
                            assd_val = m.get("assd_mean", float("nan"))
                            lines.append(
                                f"{class_name:>8} | {dice:>10.4f} | {iou:>10.4f} | {hd95:>10.2f} | {assd_val:>10.2f}"
                            )
                            if np.isfinite(dice):
                                view_dice_values.append(dice)
                    
                    if view_dice_values:
                        lines.append("-" * len(header))
                        lines.append(f"{'Mean':>8} | {np.mean(view_dice_values):>10.4f}")
        
        return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================

def evaluate_predictions(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    views: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    class_ids: Optional[Sequence[int]] = None,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    print_table: bool = True,
) -> Dict:
    """
    Evaluate a batch of predictions.
    
    Convenience function that creates an aggregator, runs evaluation,
    and returns results.
    
    Args:
        preds: [B, H, W] or list of [H, W] predictions
        targets: [B, H, W] or list of [H, W] ground truths
        views: Optional view labels
        datasets: Optional dataset labels
        class_ids: Classes to evaluate (default: all foreground)
        voxel_spacing: Pixel spacing for distance metrics
        print_table: Whether to print results table
    
    Returns:
        Full metrics dictionary
    """
    aggregator = MetricAggregator()
    
    if isinstance(preds, (list, tuple)):
        for i, (p, t) in enumerate(zip(preds, targets)):
            view = views[i] if views else None
            dataset = datasets[i] if datasets else None
            aggregator.update(p, t, view=view, dataset=dataset, 
                            class_ids=class_ids, voxel_spacing=voxel_spacing)
    else:
        aggregator.update_batch(
            preds, targets,
            views=views, datasets=datasets,
            class_ids=class_ids, voxel_spacing=voxel_spacing,
        )
    
    if print_table:
        print(aggregator.format_table())
    
    return aggregator.to_dict()


def compute_dice_only(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 6,
    ignore_background: bool = True,
) -> Dict[str, float]:
    """
    Fast dice-only computation for validation logging.
    
    Doesn't use medpy, just computes dice directly.
    
    Args:
        pred: [H, W] predicted labels
        target: [H, W] ground truth labels
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0
    
    Returns:
        Dict mapping class_name -> dice value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    start_class = 1 if ignore_background else 0
    
    results = {}
    for cid in range(start_class, num_classes):
        pred_mask = (pred == cid)
        target_mask = (target == cid)
        
        intersection = (pred_mask & target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            dice = 1.0  # Both empty
        else:
            dice = 2.0 * intersection / union
        
        class_name = ID_TO_LABEL.get(cid, f"class_{cid}")
        results[class_name] = float(dice)
    
    # Add mean
    fg_dices = [v for k, v in results.items() if k != "BG"]
    if fg_dices:
        results["mean"] = float(np.mean(fg_dices))
    
    return results


def compute_batch_dice(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int = 6,
    ignore_background: bool = True,
) -> Dict[str, float]:
    """
    Compute mean dice across a batch.
    
    Args:
        preds: [B, H, W] predicted labels
        targets: [B, H, W] ground truth labels
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0
    
    Returns:
        Dict with mean dice per class and overall mean
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    batch_size = preds.shape[0]
    
    all_dices = defaultdict(list)
    for i in range(batch_size):
        sample_dice = compute_dice_only(
            preds[i], targets[i], 
            num_classes=num_classes,
            ignore_background=ignore_background,
        )
        for k, v in sample_dice.items():
            all_dices[k].append(v)
    
    return {k: float(np.mean(v)) for k, v in all_dices.items()}
