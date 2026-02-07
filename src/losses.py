"""
Loss functions for P2Echo-new.

Ported from original VoxTell losses.py with valid_mask support.
Uses binary per-prompt paradigm: outputs are [B, N, H, W] sigmoids.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def nnunet_deep_supervision_weights(n_outputs: int) -> List[float]:
    """
    nnU-Net default style: weights proportional to 1/(2^i), normalized to sum=1.
    i=0 is the highest resolution output.
    """
    if n_outputs <= 0:
        raise ValueError("n_outputs must be > 0")
    w = [1.0 / (2.0**i) for i in range(n_outputs)]
    s = sum(w)
    return [x / s for x in w]


class DiceBCELoss(nn.Module):
    """
    Binary Dice + BCE for logits.

    Expects:
      logits:  [B, N, H, W] (or [B, N, ...])
      targets: same shape, float in {0,1}
      valid_mask: [B, N] optional mask to ignore certain prompts
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
        
        # Important for mixed precision stability: compute loss in fp32.
        logits = logits.float()
        targets = targets.float()
        
        if valid_mask is not None:
            if valid_mask.ndim != 2:
                raise ValueError(f"valid_mask must be [B,N], got {tuple(valid_mask.shape)}")
            if valid_mask.shape[0] != logits.shape[0] or valid_mask.shape[1] != logits.shape[1]:
                raise ValueError(
                    f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs logits {tuple(logits.shape)}"
                )
            valid_mask = valid_mask.float()

        dims = tuple(range(2, logits.ndim))

        # BCE (per-sample, per-prompt)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dims)  # [B,N]

        # Dice (soft) (per-sample, per-prompt)
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dims)
        fp = (probs * (1.0 - targets)).sum(dims)
        fn = ((1.0 - probs) * targets).sum(dims)
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        dice_loss = 1.0 - dice  # [B,N]

        loss_bn = dice_loss + bce  # [B,N]

        if valid_mask is None:
            return loss_bn.mean()

        loss_bn = loss_bn * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        return loss_bn.sum() / denom


class DeepSupervisionDiceBCELoss(nn.Module):
    """
    Applies Dice+BCE at all deep supervision outputs using nnU-Net-style weights.

    outputs: list of logits tensors, highest resolution first, each [B,N,H_i,W_i]
    targets: binary targets [B,N,H,W] at full resolution
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.base = DiceBCELoss(smooth=smooth)

    def forward(
        self, 
        outputs: Sequence[torch.Tensor], 
        targets: torch.Tensor, 
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list/tuple of tensors")

        weights = nnunet_deep_supervision_weights(len(outputs))

        loss = 0.0
        for w, out in zip(weights, outputs):
            if out.ndim != targets.ndim:
                raise ValueError(
                    f"Deep supervision output ndim mismatch: out {out.ndim} vs targets {targets.ndim}"
                )
            tgt = F.interpolate(targets, size=out.shape[-2:], mode="nearest")
            loss = loss + (float(w) * self.base(out, tgt, valid_mask=valid_mask))
        return loss


# =============================================================================
# Multi-class Dice + CE
# =============================================================================

class DiceCELoss(nn.Module):
    """
    Multi-class Dice + Cross-Entropy loss for categorical segmentation.

    Expects:
      logits:  [B, C, H, W] raw logits (C = num_classes)
      targets: [B, H, W] categorical labels in {0, ..., C-1}

    All classes are always supervised. If a class is absent in the image,
    the model is penalized for predicting it (false positives). The text
    prompts tell the model which classes to segment and which not to.
    """

    def __init__(self, smooth: float = 1e-5, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,  # Accept and ignore extra args (e.g. valid_mask) for API compat
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.long()
        C = logits.shape[1]

        # Cross-Entropy (per-sample)
        ce = F.cross_entropy(logits, targets, reduction="none").mean(dim=(1, 2))  # [B]

        # Soft Dice (per-class, per-sample)
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_oh = F.one_hot(targets, num_classes=C).float().permute(0, 3, 1, 2)  # [B, C, H, W]

        tp = (probs * targets_oh).sum(dim=(2, 3))           # [B, C]
        fp = (probs * (1.0 - targets_oh)).sum(dim=(2, 3))   # [B, C]
        fn = ((1.0 - probs) * targets_oh).sum(dim=(2, 3))   # [B, C]
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        dice_loss = (1.0 - dice).mean(dim=1)  # [B], averaged over classes

        # Combine
        loss = self.ce_weight * ce + self.dice_weight * dice_loss  # [B]
        return loss.mean()


class DeepSupervisionDiceCELoss(nn.Module):
    """
    Applies Dice+CE at all deep supervision outputs using nnU-Net-style weights.

    outputs: list of logits tensors, highest resolution first, each [B, C, H_i, W_i]
    targets: categorical targets [B, H, W] at full resolution (values 0..C-1)
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.base = DiceCELoss(smooth=smooth)

    def forward(
        self,
        outputs: Sequence[torch.Tensor],
        targets: torch.Tensor,
        **kwargs,  # Accept and ignore extra args for API compat
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list/tuple of tensors")

        weights = nnunet_deep_supervision_weights(len(outputs))

        loss = 0.0
        for w, out in zip(weights, outputs):
            tgt = F.interpolate(
                targets.unsqueeze(1).float(), size=out.shape[-2:], mode="nearest"
            ).squeeze(1).long()
            loss = loss + (float(w) * self.base(out, tgt))
        return loss


# =============================================================================
# BoundaryDoU Loss (from CENet)
# =============================================================================

class BoundaryDoULoss(nn.Module):
    """
    Boundary-aware Degree of Union loss for binary segmentation.
    
    Copied from CENet codebase. Adapts the loss weight based on boundary pixels.
    """
    
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def _adaptive_size(self, score, target):
        """Compute boundary-aware loss weight alpha and DoU loss."""
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        # Detect boundary pixels using convolution
        Y = torch.conv2d(
            target.unsqueeze(0).unsqueeze(0).float(),
            kernel.unsqueeze(0).unsqueeze(0).to(target.device),
            padding=1
        ).squeeze()
        Y = Y * target
        Y[Y == 5] = 0  # Non-boundary interior pixels
        
        C = torch.count_nonzero(Y)  # Boundary pixel count
        S = torch.count_nonzero(target)  # Total foreground pixels
        
        # Adaptive alpha based on boundary ratio
        alpha = 1 - (C + self.smooth) / (S + self.smooth)
        alpha = 2 * alpha - 1
        alpha = min(alpha.item() if isinstance(alpha, torch.Tensor) else alpha, 0.8)
        
        # DoU loss with adaptive weighting
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (z_sum + y_sum - 2 * intersect + self.smooth) / \
               (z_sum + y_sum - (1 + alpha) * intersect + self.smooth)
        
        return loss

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute BoundaryDoU loss.
        
        Args:
            logits: [B, N, H, W] logits
            targets: [B, N, H, W] binary targets
            valid_mask: [B, N] optional mask to ignore certain prompts
        """
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        
        losses = []
        for b in range(logits.shape[0]):
            for n in range(logits.shape[1]):
                if valid_mask is not None and valid_mask[b, n] < 0.5:
                    continue
                # Only compute loss for non-empty targets
                if targets[b, n].sum() > 0:
                    losses.append(self._adaptive_size(probs[b, n], targets[b, n]))
                else:
                    # For empty targets, use simple BCE-like loss
                    losses.append(probs[b, n].mean())
        
        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()


class DeepSupervisionBoundaryDoULoss(nn.Module):
    """
    Applies BoundaryDoU at all deep supervision outputs using nnU-Net-style weights.

    outputs: list of logits tensors, highest resolution first, each [B,N,H_i,W_i]
    targets: binary targets [B,N,H,W] at full resolution
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.base = BoundaryDoULoss(smooth=smooth)

    def forward(
        self, 
        outputs: Sequence[torch.Tensor], 
        targets: torch.Tensor, 
        valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            raise ValueError("outputs must be a non-empty list/tuple of tensors")

        weights = nnunet_deep_supervision_weights(len(outputs))

        loss = 0.0
        for w, out in zip(weights, outputs):
            if out.ndim != targets.ndim:
                raise ValueError(
                    f"Deep supervision output ndim mismatch: out {out.ndim} vs targets {targets.ndim}"
                )
            tgt = F.interpolate(targets, size=out.shape[-2:], mode="nearest")
            loss = loss + (float(w) * self.base(out, tgt, valid_mask=valid_mask))
        return loss

