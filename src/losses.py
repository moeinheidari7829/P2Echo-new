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
# Multi-class variants (for reference, not used in binary per-prompt paradigm)
# =============================================================================

class DiceCELoss(nn.Module):
    """
    Dice + CE for logits.

    Expects:
      logits:  [B, N, H, W] (or [B, N, ...])
      targets: same shape, categorical
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
        # Important for mixed precision stability: compute loss in fp32.
        logits = logits.float()
        if targets.ndim == logits.ndim and targets.shape[1] == logits.shape[1]:
            targets = targets.argmax(dim=1)
        targets = targets.long()
        
        if targets.numel() > 0:
            max_t = int(targets.max().item())
            if max_t >= logits.shape[1]:
                if max_t == logits.shape[1]:
                    zeros = torch.zeros(
                        (logits.shape[0], 1, *logits.shape[2:]), device=logits.device, dtype=logits.dtype
                    )
                    logits = torch.cat([zeros, logits], dim=1)
                    if valid_mask is not None:
                        valid_mask = F.pad(valid_mask, (1, 0), value=0.0)
                else:
                    raise ValueError(
                        f"targets has label id {max_t} but logits has only {logits.shape[1]} classes"
                    )
        
        if valid_mask is not None:
            if valid_mask.ndim != 2:
                raise ValueError(f"valid_mask must be [B,N], got {tuple(valid_mask.shape)}")
            if valid_mask.shape[0] != logits.shape[0] or valid_mask.shape[1] != logits.shape[1]:
                raise ValueError(
                    f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs logits {tuple(logits.shape)}"
                )
            valid_mask = valid_mask.float()

        dims = tuple(range(1, targets.ndim))
        # CE (per-sample, per-prompt)
        ce = F.cross_entropy(logits, targets, reduction="none").mean(dims)

        targets_oh = F.one_hot(targets, num_classes=logits.shape[1]).float().permute(0, 3, 1, 2)

        dims = tuple(range(2, targets_oh.ndim))
        # Dice (soft) (per-sample, per-prompt)
        probs = F.softmax(logits, dim=1)
        tp = (probs * targets_oh).sum(dims)
        fp = (probs * (1.0 - targets_oh)).sum(dims)
        fn = ((1.0 - probs) * targets_oh).sum(dims)
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        dice_loss = 1.0 - dice  # [B,N]

        loss_ce = dice_loss + ce.view(-1, 1)  # [B,N]

        if valid_mask is None:
            return loss_ce.mean()

        loss_ce = loss_ce * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        return loss_ce.sum() / denom


class DeepSupervisionDiceCELoss(nn.Module):
    """
    Applies Dice+CE at all deep supervision outputs using nnU-Net-style weights.

    outputs: list of logits tensors, highest resolution first, each [B,N,H_i,W_i]
    targets: categorical targets [B,H,W] at full resolution
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.base = DiceCELoss(smooth=smooth)

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
            # Handle both [B,H,W] and [B,C,H,W] style targets.
            if targets.ndim == 3:
                tgt = F.interpolate(targets.unsqueeze(1).float(), size=out.shape[-2:], mode="nearest").squeeze(1)
            else:
                tgt = F.interpolate(targets.float(), size=out.shape[-2:], mode="nearest")
            tgt = tgt.long()
            loss = loss + (float(w) * self.base(out, tgt, valid_mask=valid_mask))
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

