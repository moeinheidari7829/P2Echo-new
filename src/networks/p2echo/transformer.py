"""
Text-conditioned cross-attention transformer for P2Echo.

This module enables text-guided segmentation by letting text prompt embeddings
attend to image features, producing mask embeddings that guide the decoder.

Optimized architecture (vs original P2Echo):
- Query dim: 384 (vs 864 original)
- 3 layers (vs 6 original)  
- 6 heads (vs 8 original)
- No self-attention between prompts (only 6 prompts, minimal benefit)

Based on:
- Original P2Echo (voxtell/model/transformer.py)
- DETR transformer decoder (Facebook Research)
"""

import copy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for text-image cross-attention.
    
    Text prompt embeddings serve as queries, attending to image features
    from the encoder bottleneck to produce mask embeddings.
    
    Args:
        decoder_layer: Single decoder layer to be cloned
        num_layers: Number of decoder layers (default: 3)
        norm: Final normalization layer
        return_intermediate: Whether to return all intermediate outputs
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 3,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Forward pass through all decoder layers.
        
        Args:
            tgt: Text queries [N, B, C] where N=num_prompts (e.g., 6)
            memory: Image features [HW, B, C] from encoder bottleneck (e.g., [64, B, 384])
            tgt_mask: Attention mask for target self-attention (unused)
            memory_mask: Attention mask for cross-attention
            tgt_key_padding_mask: Padding mask for target keys
            memory_key_padding_mask: Padding mask for memory keys
            pos: Positional embeddings for memory [HW, B, C]
            query_pos: Positional embeddings for queries (unused)
            
        Returns:
            Tuple of (output, attention_weights_list):
            - output: Refined text-image fused features [N, B, C]
            - attention_weights_list: Cross-attention weights from each layer
        """
        output = tgt
        intermediate = []
        attn_layers = []
        
        for layer in self.layers:
            output, attn_weights = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            attn_layers.append(attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, attn_layers


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with cross-attention and FFN.
    
    Architecture (pre-norm style):
        1. LayerNorm → Cross-Attention (text queries attend to image features)
        2. LayerNorm → Feed-Forward Network
    
    Note: No self-attention between prompts. With only 6 prompts representing
    distinct anatomical structures, self-attention adds parameters without
    proportional benefit.
    
    Args:
        d_model: Model dimension (default: 384)
        nhead: Number of attention heads (default: 6, so 64 dim per head)
        dim_feedforward: FFN hidden dimension (default: 1024)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        
        # Cross-attention: text queries attend to image features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional embeddings to tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pre-norm forward pass.
        
        Args:
            tgt: Target sequence (text queries) [N, B, C]
            memory: Memory (image features) [HW, B, C]
            pos: Positional embeddings for memory [HW, B, C]
            query_pos: Positional embeddings for queries (optional)
            
        Returns:
            Tuple of (output, attention_weights):
            - output: [N, B, C]
            - attention_weights: [B, N, HW] attention map
        """
        # Cross-attention with pre-norm
        # Q = text queries, K/V = image features with position
        tgt2 = self.norm1(tgt)
        tgt2, attn_weights = self.cross_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        
        # Feed-forward network with pre-norm
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt, attn_weights


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return activation function by name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ============================================================================
# Convenience function to build the full transformer
# ============================================================================

def build_text_image_transformer(
    d_model: int = 384,
    nhead: int = 6,
    num_layers: int = 3,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
) -> TransformerDecoder:
    """
    Build the text-image cross-attention transformer.
    
    Args:
        d_model: Query/key/value dimension (default: 384)
        nhead: Number of attention heads (default: 6)
        num_layers: Number of decoder layers (default: 3)
        dim_feedforward: FFN hidden dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        
    Returns:
        TransformerDecoder instance
        
    Example:
        >>> transformer = build_text_image_transformer()
        >>> # text_queries: [6, B, 384], image_features: [64, B, 384]
        >>> mask_embed, attn = transformer(text_queries, image_features, pos=pos_embed)
        >>> # mask_embed: [6, B, 384] - one embedding per prompt
    """
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
    )
    return TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_layers,
        norm=nn.LayerNorm(d_model),
    )
