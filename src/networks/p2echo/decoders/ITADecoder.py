"""
DyITA Decoder for P2Echo.

A 4-stage U-Net-style decoder that uses ITABlock (Dynamic Image-Text Alignment)
for native text injection inside each decoder block, replacing the post-hoc
_inject_text pattern of CENetDecoder.

Architecture:
    Stage 4 (bottleneck):  CFAModule (text-unaware, 512ch, 8×8)
                           — small spatial size makes routing overhead dominant;
                             standard CFA block is more efficient here.
    Stage 3 (320ch, 16×16): ITABlock with DyITA cross-attention to text
    Stage 2 (128ch, 32×32): ITABlock with DyITA cross-attention to text
    Stage 1 (64ch,  64×64): ITABlock with DyITA cross-attention to text

Features:
    - Native text fusion inside blocks (not post-hoc)
    - Optional dual injection: ITABlock + post-hoc _inject_text for A/B testing
    - Optional deep supervision heads at intermediate stages
    - EUCB upsampling (same as CENetDecoder)
    - Additive skip connections from encoder

Mitigations applied:
    - Bottleneck (dec4) uses CFAModule to avoid routing overhead at tiny spatial size
    - Multi-head DyITA (n_heads=2) for prompt-region specialization
    - STE routing throughout for differentiable argmax
    - γ=3.0, λ=0.01 initialization
    - DWC(V) residual in DyITA for local spatial bias
    - Gated ConvFFN with DWConv for local mixing
    - LayerScale + DropPath for training stability
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.cfam import CFAModule
from ..modules.dyita import ITABlock
from ..modules.blocks import UpConv, UpTConv, UpRb, EUCB


class DyITADecoder(nn.Module):
    """
    Decoder with Dynamic Image-Text Alignment blocks.

    Args:
        channels: channel dimensions [deepest, ..., shallowest] (default: [512, 320, 128, 64]).
        up_block: upsampling block type ('eucb', 'uprb', 'upcn', 'uptc').
        num_classes: number of output segmentation classes.
        n_heads: DyITA attention heads per stage (list of 3, for stages 1-3).
        n_projectors: number of routable projector banks.
        n_kernel_factors: number of learnable γ factors.
        n_diff_factors: number of learnable λ factors.
        ffn_ratio: FFN expansion ratio.
        drop_path_rate: stochastic depth rate.
        dual_injection: if True, apply post-hoc _inject_text *after* ITABlock too
                        (for ablation: native + post-hoc fusion).
        deep_supervision: if True, return multi-scale outputs for DS loss.
    """

    def __init__(
        self,
        channels: list = [512, 320, 128, 64],
        up_block: str = "eucb",
        num_classes: int = 6,
        # DyITA hyperparameters
        n_heads: list = [2, 2, 2],
        n_projectors: int = 3,
        n_kernel_factors: int = 9,
        n_diff_factors: int = 9,
        ffn_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        gamma_init: float = 3.0,
        lambda_init: float = 0.01,
        # Ablation / supervision flags
        dual_injection: bool = False,
        deep_supervision: bool = True,
        writer=None,
    ):
        super().__init__()

        assert up_block in ["uprb", "eucb", "upcn", "uptc"], f"Invalid up_block: {up_block}"
        assert len(n_heads) == 3, "n_heads must have 3 entries (for stages 1-3)"

        self.num_classes = num_classes
        self.dual_injection = dual_injection
        self.deep_supervision = deep_supervision
        self.writer = writer

        # --- Upsampling block factory ---
        up_ks = 3
        if up_block == "uprb":
            up_factory = partial(UpRb, kernel_size=up_ks, scale_factor=2)
        elif up_block == "eucb":
            up_factory = partial(EUCB, kernel_size=up_ks, stride=up_ks // 2, activation="leakyrelu")
        elif up_block == "upcn":
            up_factory = partial(UpConv, kernel_size=up_ks, stride=1, activation="leakyrelu")
        elif up_block == "uptc":
            up_factory = partial(UpTConv, kernel_size=up_ks, stride=2, activation="leakyrelu")
        else:
            raise ValueError(f"Invalid up_block: {up_block}")

        # --- Stage 4 (bottleneck): CFAModule — no text injection ---
        # Mitigation: small spatial size (8×8 = 64 tokens) makes routing overhead
        # dominant relative to benefit; standard CFA is more efficient here.
        mca_rates_bottleneck = [1, 2, 2]
        self.dec4 = CFAModule(
            embed_dims=channels[0],
            ffn_ratio=4,
            drop_rate=0,
            drop_path_rate=0,
            act_type="GELU",
            norm_type="BN",
            init_value=1e-6,
            attn_channel_split=[1, 3, 4],
            attn_act_type="SiLU",
            mca_rates=mca_rates_bottleneck,
        )

        # --- Upsampling layers ---
        self.up3 = up_factory(in_channels=channels[0], out_channels=channels[1])
        self.up2 = up_factory(in_channels=channels[1], out_channels=channels[2])
        self.up1 = up_factory(in_channels=channels[2], out_channels=channels[3])

        # --- Stages 3, 2, 1: ITABlock with native text injection ---
        ita_factory = partial(
            ITABlock,
            ffn_ratio=ffn_ratio,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            init_value=0.1,  # Higher init for faster text gradient propagation
            n_projectors=n_projectors,
            n_kernel_factors=n_kernel_factors,
            n_diff_factors=n_diff_factors,
            gamma_init=gamma_init,
            lambda_init=lambda_init,
        )
        self.dec3 = ita_factory(embed_dim=channels[1], n_heads=n_heads[0])
        self.dec2 = ita_factory(embed_dim=channels[2], n_heads=n_heads[1])
        self.dec1 = ita_factory(embed_dim=channels[3], n_heads=n_heads[2])

        # --- Optional dual injection (post-hoc _inject_text after ITABlock) ---
        if self.dual_injection:
            self.inject_convs = nn.ModuleList()
            for ch in channels[1:]:
                self.inject_convs.append(nn.Sequential(
                    nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False),
                    nn.GroupNorm(max(1, ch // 16), ch),
                    nn.GELU(),
                ))

        # --- Output head ---
        self.output = nn.Conv2d(
            in_channels=channels[3],
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

        # --- Optional deep supervision heads at stages 2 and 3 ---
        if self.deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(channels[1], num_classes, kernel_size=1, bias=False),  # stage 3 (16×16)
                nn.Conv2d(channels[2], num_classes, kernel_size=1, bias=False),  # stage 2 (32×32)
            ])

    def _inject_text_posthoc(self, y, mask_embed, stage_idx):
        """
        Post-hoc text injection (same as CENetDecoder._inject_text).
        Only used when dual_injection=True.

        Args:
            y: [B, C, H, W] decoder feature map.
            mask_embed: [B, N, C] projected text embeddings.
            stage_idx: 1-based index in {1, 2, 3}.
        """
        attn = torch.einsum("b c h w, b n c -> b n h w", y, mask_embed)
        text_feat = torch.einsum("b n h w, b n c -> b c h w", attn.sigmoid(), mask_embed)
        return self.inject_convs[stage_idx - 1](torch.cat([y, text_feat], dim=1))

    def forward(
        self,
        features: list,
        inject_mask_embeds: list,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            features: encoder outputs [f1, f2, f3, f4] shallow→deep
                f1: [B, 64,  H/4,  W/4 ]
                f2: [B, 128, H/8,  W/8 ]
                f3: [B, 320, H/16, W/16]
                f4: [B, 512, H/32, W/32]
            inject_mask_embeds: list of 3 tensors [B, N, C_stage] for stages 1-3
                [0]: [B, N, 320]  for stage 3 (dec3)
                [1]: [B, N, 128]  for stage 2 (dec2)
                [2]: [B, N, 64]   for stage 1 (dec1)

        Returns:
            if deep_supervision:
                Tuple of (main_logits, ds_stage3, ds_stage2)
                Each at [B, num_classes, H, W] (interpolated to input size)
            else:
                [B, num_classes, H, W] segmentation logits
        """
        f1, f2, f3, x = features
        skips_3, skips_2, skips_1 = f3, f2, f1

        # --- Stage 4: bottleneck (text-unaware CFAModule) ---
        d4 = self.dec4(x)

        # --- Stage 3: ITABlock with native text injection ---
        d3 = self.up3(d4)
        d3 = d3 + skips_3
        d3 = self.dec3(d3, inject_mask_embeds[0])  # native DyITA fusion
        if self.dual_injection:
            d3 = self._inject_text_posthoc(d3, inject_mask_embeds[0], stage_idx=1)

        # Deep supervision output at stage 3
        if self.deep_supervision:
            ds3 = self.ds_heads[0](d3)

        # --- Stage 2: ITABlock with native text injection ---
        d2 = self.up2(d3)
        d2 = d2 + skips_2
        d2 = self.dec2(d2, inject_mask_embeds[1])  # native DyITA fusion
        if self.dual_injection:
            d2 = self._inject_text_posthoc(d2, inject_mask_embeds[1], stage_idx=2)

        # Deep supervision output at stage 2
        if self.deep_supervision:
            ds2 = self.ds_heads[1](d2)

        # --- Stage 1: ITABlock with native text injection ---
        d1 = self.up1(d2)
        d1 = d1 + skips_1
        d1 = self.dec1(d1, inject_mask_embeds[2])  # native DyITA fusion
        if self.dual_injection:
            d1 = self._inject_text_posthoc(d1, inject_mask_embeds[2], stage_idx=3)

        # --- Output ---
        logits = self.output(d1)
        # Get target spatial size from the shallowest encoder feature
        target_h, target_w = f1.shape[2] * 4, f1.shape[3] * 4
        logits = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)

        if self.deep_supervision:
            ds3 = F.interpolate(ds3, size=(target_h, target_w), mode="bilinear", align_corners=False)
            ds2 = F.interpolate(ds2, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return logits, ds3, ds2

        return logits
