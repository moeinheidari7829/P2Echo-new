"""
Frozen text backbone for P2Echo.

Uses Qwen3-Embedding-0.6B by default (1024-dim embeddings).
The text encoder is frozen during training - only the projection layers are trained.

Based on the original P2Echo implementation (voxtell/echo/text_encoder.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last non-padding token's hidden state.
    
    Qwen embedding models use last-token pooling (similar to GPT-style models)
    rather than [CLS] token or mean pooling.
    
    Args:
        last_hidden_states: [B, seq_len, hidden_size]
        attention_mask: [B, seq_len]
        
    Returns:
        Pooled embeddings [B, hidden_size]
    """
    # Check if left-padded (all sequences end at last position)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        # Right-padded: find the last non-padding position for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), 
            sequence_lengths
        ]


def wrap_with_instruction(text_prompts: Sequence[str]) -> list[str]:
    """
    Wrap prompts with instruction for Qwen embedding model.
    
    Qwen embedding models perform better when queries are wrapped with
    task-specific instructions.
    
    Args:
        text_prompts: List of raw text prompts
        
    Returns:
        List of instruction-wrapped prompts
    """
    instruct = 'Given an anatomical term query, retrieve the precise anatomical entity and location it represents'
    return [f'Instruct: {instruct}\nQuery: {text}' for text in text_prompts]


@dataclass(frozen=True)
class _CacheKey:
    """Cache key for prompt embeddings."""
    prompt: str
    device: str


class FrozenTextBackbone(nn.Module):
    """
    Frozen text encoder using Qwen3-Embedding.
    
    Features:
    - Instruction-wrapped prompts for better embedding quality
    - Last-token pooling (Qwen-style)
    - L2-normalized embeddings for stability
    - Caching to avoid recomputation of same prompts
    
    Args:
        model_name: HuggingFace model name (default: Qwen/Qwen3-Embedding-0.6B)
        max_text_length: Maximum token length for prompts
        use_instruction: Whether to wrap prompts with instruction
        
    Example:
        >>> text_encoder = FrozenTextBackbone()
        >>> prompts = ["Segment the left ventricle", "Segment the myocardium"]
        >>> embeddings = text_encoder.embed_prompts(prompts, device)
        >>> # embeddings: [2, 1024]
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_text_length: int = 128,
        use_instruction: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.max_text_length = int(max_text_length)
        self.use_instruction = bool(use_instruction)

        # Load tokenizer and model (model_name can be HF id or local path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Get embedding dimension from model config
        # Qwen3-Embedding-0.6B: hidden_size=1024
        # Qwen3-Embedding-4B: hidden_size=2560
        self.embedding_dim = int(self.model.config.hidden_size)
        
        # Cache for prompt embeddings (stored on CPU to save GPU memory)
        self._cache: Dict[_CacheKey, torch.Tensor] = {}

    @torch.no_grad()
    def embed_prompts(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Embed a list of prompt strings.

        Args:
            prompts: List of text prompts
            device: Target device for output tensors
            
        Returns:
            embeddings: [N, D] L2-normalized embeddings where D=1024 for 0.6B model
        """
        self.model.to(device)

        prompts = [str(p) for p in prompts]
        out: list = []
        missing: list = []
        missing_idx: list = []
        dev = str(device)
        
        # Check cache for existing embeddings
        for i, p in enumerate(prompts):
            key = _CacheKey(prompt=p, device=dev)
            if key in self._cache:
                out.append(self._cache[key])
            else:
                out.append(None)
                missing.append(p)
                missing_idx.append(i)

        # Compute embeddings for missing prompts
        if missing:
            # Optionally wrap with instruction
            to_embed = wrap_with_instruction(missing) if self.use_instruction else missing
            
            # Tokenize
            toks = self.tokenizer(
                to_embed,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            
            # Forward pass through frozen model
            enc = self.model(**toks)
            
            # Last-token pooling
            emb = last_token_pool(enc.last_hidden_state, toks["attention_mask"])
            
            # L2-normalize for stability (common practice for embedding models)
            emb = F.normalize(emb.float(), p=2, dim=-1)
            
            # Cache results (store on CPU)
            for j, p in enumerate(missing):
                key = _CacheKey(prompt=p, device=dev)
                self._cache[key] = emb[j].detach().cpu()
                out[missing_idx[j]] = self._cache[key]

        # Stack and move to target device
        stacked = torch.stack(out, dim=0).to(device)
        return stacked

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def forward(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Alias for embed_prompts for nn.Module compatibility."""
        return self.embed_prompts(prompts, device)

    def extra_repr(self) -> str:
        return f"model_name={self.model_name}, embedding_dim={self.embedding_dim}, use_instruction={self.use_instruction}"
