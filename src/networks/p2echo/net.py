"""
P2Echo: Text-conditioned echocardiography segmentation.

This is the main model that integrates:
- Encoder: PVT-v2-B2 (pretrained vision transformer)
- Text conditioning: Transformer with self-attention + cross-attention
- Decoder: CENet-style decoder with text injection

Architecture overview:
    Image [B,3,H,W] → PVT-v2-B2 → Multi-scale features
                                      ↓
    Text prompts → Qwen → [B,N,1024] → Project → [N,B,384]
                                                    ↓
    Bottleneck features [512ch] → Project → [HW,B,384]
                                                    ↓
                                    Self-Attn (prompts interact) → Cross-Attn → FFN
                                                    ↓
                                          Mask embeddings [B,N,384]
                                                    ↓
                                          Per-stage projections → [B,N,320], [B,N,128], [B,N,64]
                                                    ↓
    Multi-scale features → CENetDecoder (all N embeddings injected at stages 1-3)
                            ↓ Seg heads: Conv2d → [B, num_classes, H, W] logits
"""

from __future__ import annotations

from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

from .encoder import get_encoder2d
from .decoders import CENetDecoder, DyITADecoder, DyITADecoderNoCFA
from .transformer import TransformerDecoder, TransformerDecoderLayer


class P2Echo(nn.Module):
    """
    Text-conditioned 2D segmentation model for echocardiography.
    
    Uses text prompts to guide segmentation of cardiac structures.
    Each prompt (e.g., "Segment the left ventricle") produces a binary mask.
    
    All N prompt embeddings are injected simultaneously into the decoder, allowing
    them to interact via the attention-weighted aggregation in _inject_text.
    The decoder runs once and outputs [B, num_classes, H, W] directly.
    
    Args:
        input_channels: Number of input image channels (default: 3 for RGB)
        img_size: Input image size as (H, W) tuple (default: (256, 256))
        encoder_name: Encoder architecture (default: "pvt_v2_b2")
        pretrained_encoder: Load pretrained encoder weights
        freeze_encoder: Freeze encoder weights during training
        pretrained_dir: Directory containing pretrained weights
        text_embedding_dim: Dimension of text embeddings (1024 for Qwen-0.6B)
        query_dim: Cross-attention query dimension (default: 384)
        transformer_layers: Number of transformer decoder layers (default: 3)
        transformer_heads: Number of attention heads (default: 6)
        num_classes: Number of segmentation classes (output channels)
        decoder_depths: Number of blocks per decoder stage
        deep_supervision: Return multi-scale outputs for deep supervision loss
        drop_path_rate: Stochastic depth rate
        
    Input:
        img: [B, 3, H, W] RGB image
        text_embedding: [B, N, D] text embeddings from FrozenTextBackbone
            where N = number of prompts, D = text_embedding_dim
        
    Output:
        if deep_supervision: tuple of [B, num_classes, H, W] at multiple scales
        else: [B, num_classes, H, W] segmentation logits
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: Tuple[int, int] = (256, 256),
        # Encoder config
        encoder_name: str = "pvt_v2_b2",
        pretrained_encoder: bool = True,
        freeze_encoder: bool = False,
        pretrained_dir: str = ".",
        # Text conditioning config
        text_embedding_dim: int = 1024,  # Qwen3-Embedding-0.6B
        query_dim: int = 256,            # Standard (SAM, Mask2Former, DETR)
        transformer_layers: int = 3,     # Lightweight prompt decoder
        transformer_heads: int = 8,      # 32 dim/head
        transformer_ffn_dim: int = 1024,  # 4x query_dim (standard ratio)
        # Decoder config
        num_classes: int = 6,
        decoder_type: str = "cenet",  # "cenet", "ita", or "ita_nocfa" ("dyITA_NoCFA" alias)
        decoder_depths: Tuple[int, ...] = (4, 2, 1, 1),
        decoder_embed_dim: int = 64,
        deep_supervision: bool = True,
        drop_path_rate: float = 0.1,
        # DyITA decoder hyperparameters (used when decoder_type in {"ita", "ita_nocfa"})
        ita_n_heads: Tuple[int, ...] = (2, 2, 2),
        ita_n_projectors: int = 3,
        ita_n_kernel_factors: int = 9,
        ita_n_diff_factors: int = 9,
        ita_ffn_ratio: float = 4.0,
        ita_gamma_init: float = 3.0,
        ita_lambda_init: float = 0.01,
        ita_dual_injection: bool = False,
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.query_dim = query_dim
        self.decoder_embed_dim = decoder_embed_dim
        if decoder_type == "dyITA_NoCFA":
            decoder_type = "ita_nocfa"
        decoder_type = str(decoder_type).lower()
        if decoder_type not in {"cenet", "ita", "ita_nocfa"}:
            raise ValueError(
                f"Invalid decoder_type={decoder_type!r}. "
                "Choose from {'cenet', 'ita', 'ita_nocfa'} (or alias 'dyITA_NoCFA')."
            )
        self.decoder_type = decoder_type
        
        # =====================================================================
        # Encoder: PVT-v2-B2
        # =====================================================================
        self.encoder, encoder_channels = get_encoder2d(
            input_channels=input_channels,
            encoder=encoder_name,
            pretrain=pretrained_encoder,
            freeze_bb=freeze_encoder,
            base_ptdir=pretrained_dir,
        )
        # encoder_channels = [512, 320, 128, 64] (deepest to shallowest)
        # PVT-v2 forward returns: [stage1, stage2, stage3, stage4] 
        #                       = [(B,64,H/4,W/4), (B,128,H/8,W/8), (B,320,H/16,W/16), (B,512,H/32,W/32)]
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = encoder_channels[0]  # 512
        
        # =====================================================================
        # Text Conditioning: Cross-Attention Transformer
        # =====================================================================
        # Project bottleneck features to query dimension
        self.project_bottleneck = nn.Linear(self.bottleneck_channels, query_dim)
        
        # Project text embeddings to query dimension
        self.project_text = nn.Linear(text_embedding_dim, query_dim)
        
        # 2D positional encoding for bottleneck spatial features
        # For 256x256 input with PVT-v2: bottleneck is 8x8 (256/32)
        h, w = img_size[0] // 32, img_size[1] // 32
        self._init_pos_embed(h, w, query_dim)
        
        # Transformer cross-attention decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=query_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(query_dim),
        )
        
        # Mask embedding projections for text injection.
        # - cenet / ita: projections for stages 3->1 = [320, 128, 64]
        # - ita_nocfa: projections for stages 4->1 = [512, 320, 128, 64]
        decoder_in_channels = encoder_channels  # deepest -> shallowest
        self.inject_mask_projs = nn.ModuleList()
        inject_proj_channels = (
            decoder_in_channels if self.decoder_type == "ita_nocfa"
            else decoder_in_channels[1:]
        )
        for ch in inject_proj_channels:
            self.inject_mask_projs.append(nn.Sequential(
                nn.Linear(query_dim, query_dim // 2),
                nn.GELU(),
                nn.Linear(query_dim // 2, int(ch)),
            ))
        
        # =====================================================================
        # Decoder: CENet-style or DyITA decoder
        # =====================================================================
        if self.decoder_type == "ita":
            self.decoder = DyITADecoder(
                channels=decoder_in_channels,
                up_block='eucb',
                num_classes=num_classes,
                n_heads=list(ita_n_heads),
                n_projectors=ita_n_projectors,
                n_kernel_factors=ita_n_kernel_factors,
                n_diff_factors=ita_n_diff_factors,
                ffn_ratio=ita_ffn_ratio,
                drop_path_rate=drop_path_rate,
                gamma_init=ita_gamma_init,
                lambda_init=ita_lambda_init,
                dual_injection=ita_dual_injection,
                deep_supervision=deep_supervision,
            )
        elif self.decoder_type == "ita_nocfa":
            self.decoder = DyITADecoderNoCFA(
                channels=decoder_in_channels,
                up_block='eucb',
                num_classes=num_classes,
                n_heads=list(ita_n_heads),
                n_projectors=ita_n_projectors,
                n_kernel_factors=ita_n_kernel_factors,
                n_diff_factors=ita_n_diff_factors,
                ffn_ratio=ita_ffn_ratio,
                drop_path_rate=drop_path_rate,
                gamma_init=ita_gamma_init,
                lambda_init=ita_lambda_init,
                dual_injection=ita_dual_injection,
                deep_supervision=deep_supervision,
            )
        else:
            self.decoder = CENetDecoder(
                channels=decoder_in_channels, 
                scale_factors=[0.8,0.4],
                skip_mode='add',
                num_heads=[2,2,2],
                up_block='eucb',
                num_classes=num_classes,
                writer=None)
        
        
        self._init_weights()
    
    def _init_pos_embed(self, h: int, w: int, dim: int) -> None:
        """Initialize 2D positional embeddings for cross-attention."""
        dummy = torch.zeros(1, h, w, dim)
        pos_embed = PositionalEncoding2D(dim)(dummy)
        
        # Handle different output formats from PositionalEncoding2D
        if pos_embed.shape[-1] == dim:
            pos_hw_c = pos_embed  # [B, H, W, C]
        else:
            pos_hw_c = pos_embed.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        
        # Flatten spatial dims: [1, H, W, C] -> [HW, 1, C]
        pos_embed = rearrange(pos_hw_c, "b h w c -> (h w) b c")
        self.register_buffer("pos_embed", pos_embed)
    
    def _init_weights(self) -> None:
        """Initialize projection layers."""
        for m in [self.project_bottleneck, self.project_text]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for proj in self.inject_mask_projs:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self, 
        img: torch.Tensor, 
        text_embedding: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass.
        
        Args:
            img: [B, C, H, W] input image (C=3 for RGB)
            text_embedding: [B, N, D] text embeddings from FrozenTextBackbone
                N = number of prompts (e.g., 6 for BG + 5 cardiac structures)
                D = text embedding dimension (1024 for Qwen-0.6B)
            
        Returns:
            if deep_supervision:
                Tuple of (main_logits, ds0, ds1, ds2)
                Each is [B, num_classes, H, W] logits
            else:
                [B, num_classes, H, W] segmentation logits
        """
        # =====================================================================
        # 1. Encoder: Extract multi-scale features
        # =====================================================================
        encoder_features = self.encoder(img)
        # encoder_features: [f1, f2, f3, f4] = [(B,64,H/4,W/4), ..., (B,512,H/32,W/32)]
        
        # =====================================================================
        # 2. Text-Image Cross-Attention (with self-attention among prompts)
        # =====================================================================
        bottleneck = encoder_features[-1]  # [B, 512, H/32, W/32]
        
        # Project bottleneck: [B, 512, h, w] -> [B, h, w, 384] -> [hw, B, 384]
        bottleneck_embed = rearrange(bottleneck, "b c h w -> b h w c")
        bottleneck_embed = self.project_bottleneck(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w c -> (h w) b c")
        
        # Project text: [B, N, D] -> [N, B, 384]
        if text_embedding.ndim == 4:
            text_embedding = text_embedding.squeeze(2)
        text_embed = rearrange(text_embedding, "b n d -> n b d")
        text_embed = self.project_text(text_embed)
        
        # Self-attention among prompts + cross-attention to image features
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed,           # [N, B, 384] - queries
            memory=bottleneck_embed,  # [HW, B, 384] - keys/values
            pos=self.pos_embed,       # [HW, 1, 384] - positional encoding
        )
        # mask_embedding: [N, B, 384]
        
        # Rearrange to [B, N, query_dim] for per-stage projection
        mask_embedding = rearrange(mask_embedding, "n b c -> b n c")  # [B, N, query_dim]
        
        # Project mask embeddings for injection at decoder stages 1-3
        inject_mask_embeds = [proj(mask_embedding) for proj in self.inject_mask_projs]
        # cenet/ita: [B,N,320], [B,N,128], [B,N,64]
        # ita_nocfa: [B,N,512], [B,N,320], [B,N,128], [B,N,64]
        
        # =====================================================================
        # 3. Decoder: Single pass with all N embeddings injected together
        # =====================================================================
        return self.decoder(encoder_features, inject_mask_embeds)

    def get_encoder_params(self):
        """Get encoder parameters (for separate learning rate)."""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder parameters (for separate learning rate)."""
        return self.decoder.parameters()
    
    def get_text_conditioning_params(self):
        """Get text conditioning parameters (projections + transformer)."""
        params = []
        params.extend(self.project_bottleneck.parameters())
        params.extend(self.project_text.parameters())
        params.extend(self.transformer_decoder.parameters())
        for proj in self.inject_mask_projs:
            params.extend(proj.parameters())
        return params


def build_p2echo(
    img_size: Tuple[int, int] = (256, 256),
    num_classes: int = 6,
    pretrained_encoder: bool = True,
    pretrained_dir: str = ".",
    text_embedding_dim: int = 1024,
    deep_supervision: bool = True,
    decoder_type: str = "cenet",
    ita_dual_injection: bool = False,
) -> P2Echo:
    """
    Build P2Echo model with default configuration.
    
    Args:
        img_size: Input image size
        num_classes: Number of segmentation classes (output channels)
        pretrained_encoder: Load pretrained PVT-v2-B2 weights
        pretrained_dir: Directory containing pretrained weights
        text_embedding_dim: Text embedding dimension (1024 for Qwen-0.6B)
        deep_supervision: Enable deep supervision
        
    Returns:
        P2Echo model instance
        
    Example:
        >>> model = build_p2echo(num_classes=6, pretrained_dir="/path/to/pretrained_pth")
        >>> img = torch.randn(2, 3, 256, 256)
        >>> text_emb = torch.randn(2, 6, 1024)  # 6 prompts
        >>> out = model(img, text_emb)
        >>> # out: tuple of 4 tensors for deep supervision
        >>> # out[0]: [2, 6, 256, 256] - main output
    """
    return P2Echo(
        input_channels=3,
        img_size=img_size,
        encoder_name="pvt_v2_b2",
        pretrained_encoder=pretrained_encoder,
        freeze_encoder=False,
        pretrained_dir=pretrained_dir,
        text_embedding_dim=text_embedding_dim,
        query_dim=384,
        transformer_layers=3,
        transformer_heads=6,
        transformer_ffn_dim=1024,
        num_classes=num_classes,
        decoder_type=decoder_type,
        decoder_depths=(4, 2, 1, 1),
        decoder_embed_dim=64,
        deep_supervision=deep_supervision,
        drop_path_rate=0.1,
        ita_dual_injection=ita_dual_injection,
    )
