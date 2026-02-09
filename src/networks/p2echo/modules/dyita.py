"""
Dynamic Image-Text Alignment (DyITA) module for P2Echo.

Adapts the three core ideas from DyDiLA (Dynamic Differential Linear Attention,
arXiv 2601.13683) to perform image-text alignment inside each decoder stage:

1. Dynamic Projection Module — token-specific projectors via routed argmax (STE)
   to decouple redundancy representations Q' and K'.
2. Dynamic Measure Kernel — norm-preserving power operations with routed γ
   factors for sharper similarity measurement.
3. Token Differential Operator — removes redundant information via learnable
   per-token λ factors:  (Q̃ − λ^Q Q̃') · [(K̃ − λ^K K̃')^T V]

Adaptation for cross-modal segmentation:
- Q from image features [B, HW, C],  K/V from text mask embeddings [B, N, C]
- Multi-head support (2–4 heads) for prompt-region specialization
- DWC(V)-style depth-wise conv residual for local spatial bias
- STE (Straight-Through Estimator) for differentiable argmax routing

Components:
- DyITAModule: The core DyITA cross-attention block
- GatedConvFFN: Gated depth-wise convolutional FFN (Conv1x1 → SiLU → DWC3x3 → SiLU → gate → Conv1x1)
- ITABlock: Full decoder block (Norm → DyITA → residual → Norm → FFN → residual)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath


# =============================================================================
# Helper: Straight-Through Estimator for argmax routing
# =============================================================================

def _ste_argmax_route(scores: torch.Tensor) -> torch.Tensor:
    """
    Differentiable argmax routing via Straight-Through Estimator.

    Forward: hard one-hot selection (argmax).
    Backward: gradient flows through softmax scores.

    Args:
        scores: [*, n_choices]  raw logits from router.

    Returns:
        one_hot_ste: [*, n_choices]  one-hot in forward, soft in backward.
    """
    idx = scores.argmax(dim=-1)                            # [*]
    one_hot = F.one_hot(idx, scores.shape[-1]).float()     # [*, n_choices]
    soft = F.softmax(scores, dim=-1)                       # [*, n_choices]
    return one_hot + soft - soft.detach()                  # STE


# =============================================================================
# Dynamic Projection Module
# =============================================================================

class DynamicProjectionModule(nn.Module):
    """
    Produces standard Q, K, V plus *redundancy* representations Q', K'
    via token-specific projectors chosen by argmax routing (STE).

    For cross-modal use:
        Q, Q' come from image features  (query side)
        K, K', V come from text embeddings (key/value side)

    Args:
        img_dim: channel dim of image features (query).
        txt_dim: channel dim of text embeddings (key/value).
        head_dim: per-head dimension.
        n_heads: number of attention heads.
        n_projectors: number of routable projector banks (default: 3, "Small").
    """

    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        head_dim: int,
        n_heads: int,
        n_projectors: int = 3,
    ) -> None:
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_projectors = n_projectors

        # --- shared projections (standard) ---
        self.q_proj = nn.Linear(img_dim, inner, bias=False)
        self.k_proj = nn.Linear(txt_dim, inner, bias=False)
        self.v_proj = nn.Linear(txt_dim, inner, bias=False)

        # --- routable redundancy projectors for Q' and K' ---
        # Each bank: Linear(in_dim, inner)
        self.q_prime_banks = nn.ModuleList([
            nn.Linear(img_dim, inner, bias=False) for _ in range(n_projectors)
        ])
        self.k_prime_banks = nn.ModuleList([
            nn.Linear(txt_dim, inner, bias=False) for _ in range(n_projectors)
        ])

        # --- routers ---
        self.router_q = nn.Linear(img_dim, n_projectors, bias=False)
        self.router_k = nn.Linear(txt_dim, n_projectors, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        img_tokens: torch.Tensor,   # [B, S, img_dim]   S = H*W
        txt_tokens: torch.Tensor,   # [B, N, txt_dim]   N = num prompts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Returns Q, K, V, Q', K' each with shape [B, n_heads, seq_len, head_dim].
        """
        B, S, _ = img_tokens.shape
        _, N, _ = txt_tokens.shape

        # Standard projections
        Q = self.q_proj(img_tokens)    # [B, S, inner]
        K = self.k_proj(txt_tokens)    # [B, N, inner]
        V = self.v_proj(txt_tokens)    # [B, N, inner]

        # --- Route Q' ---
        q_scores = self.router_q(img_tokens)          # [B, S, n_P]
        q_route = _ste_argmax_route(q_scores)          # [B, S, n_P]
        # Compute all banks then mix via one-hot
        q_primes = torch.stack(
            [bank(img_tokens) for bank in self.q_prime_banks], dim=-1
        )  # [B, S, inner, n_P]
        Q_prime = (q_primes * q_route.unsqueeze(2)).sum(dim=-1)  # [B, S, inner]

        # --- Route K' ---
        k_scores = self.router_k(txt_tokens)          # [B, N, n_P]
        k_route = _ste_argmax_route(k_scores)          # [B, N, n_P]
        k_primes = torch.stack(
            [bank(txt_tokens) for bank in self.k_prime_banks], dim=-1
        )  # [B, N, inner, n_P]
        K_prime = (k_primes * k_route.unsqueeze(2)).sum(dim=-1)  # [B, N, inner]

        # Reshape to multi-head: [B, n_heads, seq, head_dim]
        def _to_heads(x: torch.Tensor) -> torch.Tensor:
            b, l, _ = x.shape
            return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        return _to_heads(Q), _to_heads(K), _to_heads(V), _to_heads(Q_prime), _to_heads(K_prime)


# =============================================================================
# Dynamic Measure Kernel
# =============================================================================

class DynamicMeasureKernel(nn.Module):
    """
    Norm-preserving power operation with per-token routed γ factors.

    φ_γ(Z) = ReLU(Z)^γ / ‖ReLU(Z)^γ‖₂ · ‖ReLU(Z)‖₂

    This adjusts the *direction* of token vectors (clustering semantically
    similar ones) while preserving the original *norm*.

    Args:
        dim: token feature dimension (head_dim for per-head application).
        n_factors: number of learnable γ factors (default: 9, "Small").
        gamma_init: initial value for all γ factors (default: 3.0).
    """

    def __init__(self, dim: int, n_factors: int = 9, gamma_init: float = 3.0) -> None:
        super().__init__()
        self.n_factors = n_factors
        # Learnable γ values — initialized to gamma_init (critical, per FLatten paper)
        self.gammas = nn.Parameter(torch.full((n_factors,), gamma_init))
        # Router: maps each token to one of n_factors kernel functions
        self.router = nn.Linear(dim, n_factors, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [..., dim]  (arbitrary leading dims, last dim = feature)

        Returns:
            Z_tilde: [..., dim]  norm-preserving power-transformed tokens.
        """
        # Route each token to a γ factor
        scores = self.router(Z.detach())               # [..., n_factors]
        route = _ste_argmax_route(scores)              # [..., n_factors]
        # Selected γ per token: [..., 1]
        gamma = (route * self.gammas.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [..., 1]
        gamma = gamma.clamp(min=1.0)  # stability: γ ≥ 1

        # Norm-preserving power operation
        # Use softplus instead of ReLU to avoid killing gradients on negative values.
        # softplus is a smooth approximation: softplus(x) = log(1 + exp(x))
        # This is critical for cross-modal alignment where features from text
        # projections may not be predominantly positive.
        Z_pos = F.softplus(Z, beta=1.0)                # [..., dim]  smooth, always > 0
        orig_norm = Z_pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        Z_pow = Z_pos.pow(gamma)                       # [..., dim]
        pow_norm = Z_pow.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        Z_tilde = Z_pow / pow_norm * orig_norm         # [..., dim]
        return Z_tilde


# =============================================================================
# Token Differential Operator
# =============================================================================

class TokenDifferentialOperator(nn.Module):
    """
    Computes:  O = (Q̃ − λ^Q Q̃') · [(K̃ − λ^K K̃')^T V]

    Per-token λ values are selected from a learnable bank via routing.
    λ initialized to 0.01 (small to prevent over-differencing early in training).

    Args:
        dim: token feature dimension (concatenated Q+Q' dim for routing, = 2*head_dim).
        n_diff_factors: number of learnable λ values (default: 9, "Small").
        lambda_init: initial value for all λ (default: 0.01).
    """

    def __init__(
        self,
        dim: int,
        n_diff_factors: int = 9,
        lambda_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_diff_factors = n_diff_factors
        # Learnable λ values — small init prevents over-differencing
        self.lambdas = nn.Parameter(torch.full((n_diff_factors,), lambda_init))
        # Routers for Q-side and K-side λ selection
        # Input dim is 2*head_dim because we route on concat(Q̃, Q̃')
        self.router_q = nn.Linear(dim, n_diff_factors, bias=False)
        self.router_k = nn.Linear(dim, n_diff_factors, bias=False)
        nn.init.xavier_uniform_(self.router_q.weight)
        nn.init.xavier_uniform_(self.router_k.weight)

    def forward(
        self,
        Q_t: torch.Tensor,       # [B, heads, S, hd]  Q̃
        Q_prime_t: torch.Tensor,  # [B, heads, S, hd]  Q̃'
        K_t: torch.Tensor,       # [B, heads, N, hd]  K̃
        K_prime_t: torch.Tensor,  # [B, heads, N, hd]  K̃'
        V: torch.Tensor,         # [B, heads, N, hd]
    ) -> torch.Tensor:
        """
        Token-wise differential paradigm (efficient: 2 matmuls).

        Returns:
            O: [B, heads, S, hd]
        """
        # Concatenate for routing
        Q_cat = torch.cat([Q_t, Q_prime_t], dim=-1)    # [B, heads, S, 2*hd]
        K_cat = torch.cat([K_t, K_prime_t], dim=-1)    # [B, heads, N, 2*hd]

        # Route λ^Q per image token
        q_scores = self.router_q(Q_cat.detach())       # [B, heads, S, n_D]
        q_route = _ste_argmax_route(q_scores)          # [B, heads, S, n_D]
        lambda_q = (q_route * self.lambdas).sum(dim=-1, keepdim=True)  # [B, heads, S, 1]

        # Route λ^K per text token
        k_scores = self.router_k(K_cat.detach())       # [B, heads, N, n_D]
        k_route = _ste_argmax_route(k_scores)          # [B, heads, N, n_D]
        lambda_k = (k_route * self.lambdas).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]

        # Differential: remove redundancy
        Q_diff = Q_t - lambda_q * Q_prime_t            # [B, heads, S, hd]
        K_diff = K_t - lambda_k * K_prime_t            # [B, heads, N, hd]

        # Linear attention: O = Q_diff · (K_diff^T V)
        # K_diff^T V: [B, heads, hd, hd]  (key-value outer product aggregated)
        KV = torch.matmul(K_diff.transpose(-2, -1), V)  # [B, heads, hd, hd]
        O = torch.matmul(Q_diff, KV)                     # [B, heads, S, hd]

        return O


# =============================================================================
# Full DyITA Module (cross-modal image-text alignment)
# =============================================================================

class DyITAModule(nn.Module):
    """
    Dynamic Image-Text Alignment module.

    Combines Dynamic Projection → Dynamic Measure Kernel → Token Differential
    Operator, adapted for cross-modal attention between image features and
    text mask embeddings.

    Args:
        img_dim: image feature channel dimension (e.g., 320, 128, 64).
        txt_dim: text embedding dimension (same as img_dim after stage projection).
        n_heads: number of attention heads (default: 2).
        n_projectors: number of routable projector banks (default: 3).
        n_kernel_factors: number of learnable γ factors (default: 9).
        n_diff_factors: number of learnable λ factors (default: 9).
        gamma_init: initial γ value (default: 3.0).
        lambda_init: initial λ value (default: 0.01).
        dropout: output dropout rate (default: 0.0).
    """

    def __init__(
        self,
        img_dim: int,
        txt_dim: Optional[int] = None,
        n_heads: int = 2,
        n_projectors: int = 3,
        n_kernel_factors: int = 9,
        n_diff_factors: int = 9,
        gamma_init: float = 3.0,
        lambda_init: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if txt_dim is None:
            txt_dim = img_dim
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.n_heads = n_heads
        self.head_dim = img_dim // n_heads
        assert img_dim % n_heads == 0, f"img_dim {img_dim} must be divisible by n_heads {n_heads}"

        # 1. Dynamic Projection
        self.dyn_proj = DynamicProjectionModule(
            img_dim=img_dim,
            txt_dim=txt_dim,
            head_dim=self.head_dim,
            n_heads=n_heads,
            n_projectors=n_projectors,
        )

        # 2. Dynamic Measure Kernel (applied per-head to Q, K, Q', K')
        self.kernel_q = DynamicMeasureKernel(self.head_dim, n_kernel_factors, gamma_init)
        self.kernel_k = DynamicMeasureKernel(self.head_dim, n_kernel_factors, gamma_init)
        self.kernel_q_prime = DynamicMeasureKernel(self.head_dim, n_kernel_factors, gamma_init)
        self.kernel_k_prime = DynamicMeasureKernel(self.head_dim, n_kernel_factors, gamma_init)

        # 3. Token Differential Operator
        self.tdo = TokenDifferentialOperator(
            dim=2 * self.head_dim,  # concat(Q̃, Q̃') dim
            n_diff_factors=n_diff_factors,
            lambda_init=lambda_init,
        )

        # 4. DWC(V) residual — provides local spatial bias (critical for segmentation)
        # Applied to image features in BCHW format
        self.dwc_v = nn.Sequential(
            nn.Conv2d(img_dim, img_dim, kernel_size=3, padding=1, groups=img_dim, bias=False),
            nn.BatchNorm2d(img_dim),
        )

        # 5. Output projection
        self.out_proj = nn.Linear(img_dim, img_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Scaling for numerical stability
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        img_feat: torch.Tensor,     # [B, C, H, W]
        txt_embed: torch.Tensor,    # [B, N, C]
    ) -> torch.Tensor:
        """
        Cross-modal DyITA attention.

        Args:
            img_feat: [B, C, H, W] decoder feature map.
            txt_embed: [B, N, C] projected text mask embeddings for this stage.

        Returns:
            out: [B, C, H, W] text-aligned image features.
        """
        B, C, H, W = img_feat.shape

        # DWC residual (local spatial bias) computed on original image features
        dwc_res = self.dwc_v(img_feat)  # [B, C, H, W]

        # Flatten spatial dims: [B, C, H, W] → [B, S, C] where S = H*W
        img_tokens = img_feat.flatten(2).transpose(1, 2)   # [B, S, C]

        # 1. Dynamic Projection → Q, K, V, Q', K'   all [B, heads, seq, hd]
        Q, K, V, Q_prime, K_prime = self.dyn_proj(img_tokens, txt_embed)

        # 2. Dynamic Measure Kernel → Q̃, K̃, Q̃', K̃'
        Q_t = self.kernel_q(Q)
        K_t = self.kernel_k(K)
        Q_prime_t = self.kernel_q_prime(Q_prime)
        K_prime_t = self.kernel_k_prime(K_prime)

        # 3. Token Differential Operator
        O = self.tdo(Q_t, Q_prime_t, K_t, K_prime_t, V)   # [B, heads, S, hd]

        # Scale
        O = O * self.scale

        # Merge heads: [B, heads, S, hd] → [B, S, C]
        O = O.transpose(1, 2).contiguous().view(B, H * W, C)

        # Output projection
        O = self.out_proj(O)
        O = self.dropout(O)

        # Reshape back to spatial: [B, S, C] → [B, C, H, W]
        O = O.transpose(1, 2).view(B, C, H, W)

        # Add DWC(V) residual for local spatial bias
        O = O + dwc_res

        return O


# =============================================================================
# Gated Convolutional FFN (matches figure: Conv1×1 → SiLU → DWC3×3 → SiLU → gate → Conv1×1)
# =============================================================================

class GatedConvFFN(nn.Module):
    """
    Gated depth-wise convolutional feed-forward network.

    Architecture (matching the user's figure):
        x → Conv1×1 (expand to 2*hidden) → SiLU
          → DWConv3×3 (on first half) → SiLU
          → gate multiply (first_half * second_half)
          → Conv1×1 (project back to embed_dim)

    The gating mechanism (SwiGLU-style split) ensures the FFN learns
    to selectively pass information, while the DWConv provides local
    spatial mixing — important for dense prediction tasks.

    Args:
        embed_dim: input/output channel dimension.
        hidden_ratio: expansion ratio for hidden dimension (default: 4).
        drop: dropout rate (default: 0.0).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * hidden_ratio)
        # Expand to 2*hidden for gating (split into value + gate branches)
        self.fc1 = nn.Conv2d(embed_dim, 2 * hidden, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=False)
        # DWConv on the value branch for local spatial mixing
        self.dwconv = nn.Conv2d(
            hidden, hidden,
            kernel_size=3, padding=1,
            groups=hidden, bias=False,
        )
        self.dwconv_bn = nn.BatchNorm2d(hidden)
        self.act2 = nn.SiLU(inplace=False)
        # Project back
        self.fc2 = nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        # Expand and split into value + gate
        x = self.fc1(x)                           # [B, 2*hidden, H, W]
        value, gate = x.chunk(2, dim=1)            # each [B, hidden, H, W]

        # Value branch: DWConv + activation
        value = self.act(value)
        value = self.dwconv(value)
        value = self.dwconv_bn(value)
        value = self.act2(value)

        # Gate branch: activation
        gate = self.act(gate)

        # Gated multiplication
        x = value * gate                           # [B, hidden, H, W]
        x = self.drop(x)

        # Project back
        x = self.fc2(x)                            # [B, C, H, W]
        x = self.drop(x)
        return x


# =============================================================================
# ITA Block: Norm → DyITA → residual → Norm → FFN → residual
# =============================================================================

class ITABlock(nn.Module):
    """
    Image-Text Alignment decoder block.

    Replaces CFAModule by natively incorporating text embeddings inside the
    block rather than as a post-hoc injection step.

    Architecture:
        x, txt → Norm(x) → DyITA(x, txt) → LayerScale → + residual
                → Norm(x) → GatedConvFFN(x)  → LayerScale → + residual

    Args:
        embed_dim: feature channel dimension.
        n_heads: number of DyITA attention heads (default: 2).
        ffn_ratio: FFN expansion ratio (default: 4.0).
        drop_rate: FFN dropout rate (default: 0.0).
        drop_path_rate: stochastic depth rate (default: 0.0).
        init_value: LayerScale initial value (default: 1e-2).
                   NOTE: 1e-6 causes text gradients to underflow to zero because
                   text information only flows through the LayerScale-gated DyITA
                   branch (unlike image features which also have the identity
                   residual). A value of 1e-2 keeps the stabilizing effect while
                   allowing meaningful text gradients at init.
        n_projectors: DyITA projector banks (default: 3).
        n_kernel_factors: DyITA γ factors (default: 9).
        n_diff_factors: DyITA λ factors (default: 9).
        gamma_init: DyITA γ initialization (default: 3.0).
        lambda_init: DyITA λ initialization (default: 0.01).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 2,
        ffn_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_value: float = 1e-2,
        # DyITA hyperparameters
        n_projectors: int = 3,
        n_kernel_factors: int = 9,
        n_diff_factors: int = 9,
        gamma_init: float = 3.0,
        lambda_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-norm layers (BatchNorm2d for BCHW tensors, matching CENet convention)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)

        # DyITA cross-modal attention
        self.dyita = DyITAModule(
            img_dim=embed_dim,
            txt_dim=embed_dim,  # text embeds are projected to match stage channels
            n_heads=n_heads,
            n_projectors=n_projectors,
            n_kernel_factors=n_kernel_factors,
            n_diff_factors=n_diff_factors,
            gamma_init=gamma_init,
            lambda_init=lambda_init,
            dropout=drop_rate,
        )

        # Gated Conv FFN
        self.ffn = GatedConvFFN(
            embed_dim=embed_dim,
            hidden_ratio=ffn_ratio,
            drop=drop_rate,
        )

        # DropPath (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # LayerScale (per-channel learnable scaling, matches CFAModule convention)
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dim, 1, 1)), requires_grad=True,
        )
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dim, 1, 1)), requires_grad=True,
        )

    def forward(
        self,
        x: torch.Tensor,             # [B, C, H, W]
        txt_embed: torch.Tensor,      # [B, N, C]
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] decoder feature map.
            txt_embed: [B, N, C] projected text mask embeddings.

        Returns:
            [B, C, H, W] text-aligned, FFN-refined features.
        """
        # DyITA branch: Norm → DyITA(x, txt) → LayerScale → + residual
        identity = x
        x_norm = self.norm1(x)
        x_att = self.dyita(x_norm, txt_embed)
        x = identity + self.drop_path(self.layer_scale_1 * x_att)

        # FFN branch: Norm → GatedConvFFN → LayerScale → + residual
        identity = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = identity + self.drop_path(self.layer_scale_2 * x_ffn)

        return x
