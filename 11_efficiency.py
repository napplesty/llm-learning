"""
================================================================================
LLM Learning Module 11: EFFICIENCY TECHNIQUES
================================================================================

Modern LLM efficiency techniques for faster training and inference:

1. Flash Attention - Memory-efficient attention computation
2. KV Cache - Cache key-value pairs for faster generation
3. Sliding Window Attention - Handle long contexts efficiently
4. ALiBi - Alternative positional encoding
5. Gradient Checkpointing - Trade compute for memory

================================================================================
ILLUSTRATION: Flash Attention
================================================================================

Standard Attention:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Q, K, V (HBM)                                                     │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────┐                                               │
    │    │ S = QK^T        │  ← Write S to HBM (O(N²) memory)             │
    │    └─────────────────┘                                               │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────┐                                               │
    │    │ P = softmax(S)  │  ← Read S, Write P to HBM                     │
    │    └─────────────────┘                                               │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────┐                                               │
    │    │ O = PV          │  ← Read P, Write O to HBM                     │
    │    └─────────────────┘                                               │
    │                                                                      │
    │    HBM reads/writes: O(N²)                                          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Flash Attention (Memory-Efficient):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Q, K, V (HBM)                                                     │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────────────────────────────────────────────────┐   │
    │    │                    SRAM (fast)                              │   │
    │    │                                                             │   │
    │    │   For each block of Q:                                      │   │
    │    │     Load Q_block, K, V blocks                               │   │
    │    │     Compute attention locally                                │   │
    │    │     Update output incrementally (online softmax trick)       │   │
    │    │                                                             │   │
    │    └─────────────────────────────────────────────────────────────┘   │
    │         │                                                            │
    │         ▼                                                            │
    │    Output (HBM)                                                      │
    │                                                                      │
    │    HBM reads/writes: O(N)  ← Linear instead of quadratic!           │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# 1. Flash Attention (Simplified Implementation)
# =============================================================================

class FlashAttention(nn.Module):
    """
    Memory-Efficient Attention using block-wise computation.

    This is a simplified implementation. Production code uses CUDA kernels.

    Key idea: Process attention in blocks to avoid materializing the full
    N×N attention matrix. Uses online softmax for numerical stability.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        block_size: Size of computation blocks (default: 64)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Online Softmax Trick:                                                    ║
    ║                                                                           ║
    ║  Instead of computing softmax over all values at once:                    ║
    ║    softmax([x1, x2, x3, x4])                                              ║
    ║                                                                           ║
    ║  We can compute incrementally:                                            ║
    ║    m_new = max(m_old, m_block)     # Track running max                   ║
    ║    d_new = d_old * exp(m_old - m_new) + exp(m_block - m_new)  # Rescale  ║
    ║    o_new = (o_old * d_old * exp(m_old - m_new) + block_out) / d_new      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_heads: int, block_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Block-wise attention computation.

        Args:
            q, k, v: (batch, heads, seq_len, d_k)
            mask: Optional causal mask

        Returns:
            output: (batch, heads, seq_len, d_k)
        """
        batch_size, num_heads, seq_len, d_k = q.shape
        block_size = min(self.block_size, seq_len)

        # Initialize output and log-sum-exp
        output = torch.zeros_like(q)
        lse = torch.full(
            (batch_size, num_heads, seq_len, 1),
            float('-inf'),
            dtype=q.dtype,
            device=q.device,
        )

        # Process query blocks
        for q_start in range(0, seq_len, block_size):
            q_end = min(q_start + block_size, seq_len)
            q_block = q[:, :, q_start:q_end, :]

            # Running max and sum for online softmax
            m = torch.full(
                (batch_size, num_heads, q_end - q_start, 1),
                float('-inf'),
                dtype=q.dtype,
                device=q.device,
            )
            l = torch.zeros(
                (batch_size, num_heads, q_end - q_start, 1),
                dtype=q.dtype,
                device=q.device,
            )
            o = torch.zeros(
                (batch_size, num_heads, q_end - q_start, d_k),
                dtype=q.dtype,
                device=q.device,
            )

            # Process key-value blocks
            for kv_start in range(0, seq_len, block_size):
                kv_end = min(kv_start + block_size, seq_len)

                k_block = k[:, :, kv_start:kv_end, :]
                v_block = v[:, :, kv_start:kv_end, :]

                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(d_k)

                # Apply mask
                if mask is not None:
                    block_mask = mask[:, :, q_start:q_end, kv_start:kv_end]
                    scores = scores.masked_fill(block_mask == 0, float('-inf'))

                # Online softmax update
                m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True)[0])
                p = torch.exp(scores - m_new)
                l_new = l * torch.exp(m - m_new) + p.sum(dim=-1, keepdim=True)
                o = o * torch.exp(m - m_new) + torch.matmul(p, v_block)

                m = m_new
                l = l_new

            # Normalize output
            output[:, :, q_start:q_end, :] = o / (l + 1e-6)
            lse[:, :, q_start:q_end, :] = m + torch.log(l + 1e-6)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self._flash_attention_forward(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


# =============================================================================
# 2. KV Cache for Efficient Generation
# =============================================================================

@dataclass
class KVCache:
    """
    Key-Value Cache for autoregressive generation.

    During generation, we compute attention over all previous tokens.
    Without caching, we'd recompute K and V for all previous tokens each step.
    With caching, we store K and V and only compute for the new token.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Generation without KV Cache (O(N²) total compute):                       ║
    ║                                                                           ║
    ║  Step 1: Compute K,V for [T1]           → Attend(T1, [T1])               ║
    ║  Step 2: Compute K,V for [T1,T2]        → Attend(T2, [T1,T2])            ║
    ║  Step 3: Compute K,V for [T1,T2,T3]     → Attend(T3, [T1,T2,T3])         ║
    ║  ...                                                                      ║
    ║                                                                           ║
    ║  Generation with KV Cache (O(N) total compute):                          ║
    ║                                                                           ║
    ║  Step 1: Compute K,V for [T1], cache    → Attend(T1, [T1])               ║
    ║  Step 2: Compute K,V for [T2], append   → Attend(T2, [cached + T2])      ║
    ║  Step 3: Compute K,V for [T3], append   → Attend(T3, [cached + T3])      ║
    ║  ...                                                                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    k_cache: torch.Tensor  # (batch, layers, heads, seq_len, d_k)
    v_cache: torch.Tensor  # (batch, layers, heads, seq_len, d_k)
    current_pos: int = 0

    @classmethod
    def create(
        cls,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        d_k: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> "KVCache":
        """Create empty KV cache."""
        k_cache = torch.zeros(
            (batch_size, num_layers, num_heads, max_seq_len, d_k),
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            (batch_size, num_layers, num_heads, max_seq_len, d_k),
            device=device,
            dtype=dtype,
        )
        return cls(k_cache, v_cache, 0)

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V values.

        Args:
            layer_idx: Which layer's cache to update
            k_new: New key tensor (batch, heads, new_seq_len, d_k)
            v_new: New value tensor (batch, heads, new_seq_len, d_k)

        Returns:
            Full K, V tensors up to current position
        """
        new_seq_len = k_new.shape[2]

        # Store new values
        self.k_cache[:, layer_idx, :, self.current_pos:self.current_pos + new_seq_len, :] = k_new
        self.v_cache[:, layer_idx, :, self.current_pos:self.current_pos + new_seq_len, :] = v_new

        # Return full cached K, V
        k_full = self.k_cache[:, layer_idx, :, :self.current_pos + new_seq_len, :]
        v_full = self.v_cache[:, layer_idx, :, :self.current_pos + new_seq_len, :]

        self.current_pos += new_seq_len

        return k_full, v_full

    def clear(self):
        """Reset cache for new generation."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_pos = 0


class CachedAttention(nn.Module):
    """
    Attention with KV Cache support for efficient generation.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for cache
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            kv_cache: Optional KV cache
            layer_idx: Layer index for cache
            use_cache: Whether to use/update cache

        Returns:
            output: Attention output
            kv_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if use_cache and kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask
        if use_cache:
            # During generation, only mask future
            mask = torch.ones(seq_len, k.shape[2], device=x.device).triu(k.shape[2] - seq_len + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out), kv_cache


# =============================================================================
# 3. Sliding Window Attention
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention for long sequences.

    Each token only attends to a window of nearby tokens, reducing
    complexity from O(N²) to O(N × W) where W is window size.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        window_size: Size of attention window
        max_seq_len: Maximum sequence length

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Full Attention vs Sliding Window:                                        ║
    ║                                                                           ║
    ║  Full (N=8):                     Sliding Window (W=3):                    ║
    ║  ┌────────────────┐              ┌────────────────┐                       ║
    ║  │1 1 1 1 1 1 1 1 │              │1 0 0 0 0 0 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │1 1 0 0 0 0 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │1 1 1 0 0 0 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │0 1 1 1 0 0 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │0 0 1 1 1 0 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │0 0 0 1 1 1 0 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │0 0 0 0 1 1 1 0 │                       ║
    ║  │1 1 1 1 1 1 1 1 │              │0 0 0 0 0 1 1 1 │                       ║
    ║  └────────────────┘              └────────────────┘                       ║
    ║  Complexity: O(N²)              Complexity: O(N × W)                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Create sliding window mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # Zero out positions outside window
        for i in range(max_seq_len):
            mask[i, :max(0, i - window_size + 1)] = 0
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


# =============================================================================
# 4. ALiBi (Attention with Linear Biases)
# =============================================================================

class ALiBiAttention(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Instead of adding positional embeddings, ALiBi adds a static bias
    to attention scores based on distance. This bias decreases linearly
    with distance, allowing the model to extrapolate to longer sequences.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  ALiBi Bias Matrix:                                                       ║
    ║                                                                           ║
    ║  bias = -m × |i - j|  where m is head-specific slope                      ║
    ║                                                                           ║
    ║  For m = 0.5:                                                             ║
    ║       pos:  0    1    2    3    4                                        ║
    ║  ┌─────────────────────────────────────┐                                  ║
    ║  │  0   -0.5  -1.0  -1.5  -2.0  │  pos 0                                 ║
    ║  │ -0.5   0   -0.5  -1.0  -1.5  │  pos 1                                 ║
    ║  │ -1.0  -0.5   0   -0.5  -1.0  │  pos 2                                 ║
    ║  │ -1.5  -1.0  -0.5   0   -0.5  │  pos 3                                 ║
    ║  │ -2.0  -1.5  -1.0  -0.5   0   │  pos 4                                 ║
    ║  └─────────────────────────────────────┘                                  ║
    ║                                                                           ║
    ║  Different heads use different slopes: m_h = 1 / 2^(8h/H)                ║
    ║  Head 0: 1/256, Head 1: 1/128, ... (geometric sequence)                  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Compute ALiBi slopes for each head
        # m_h = 1 / 2^(8h/H) for h = 0, 1, ..., H-1
        slopes = 1.0 / (2 ** (8 * torch.arange(num_heads) / num_heads))
        self.register_buffer("slopes", slopes)

        # Precompute distance matrix
        distance = torch.abs(
            torch.arange(max_seq_len).unsqueeze(0) -
            torch.arange(max_seq_len).unsqueeze(1)
        )
        self.register_buffer("distance", distance)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add ALiBi bias
        # slopes: (num_heads,), distance: (seq_len, seq_len)
        # bias: (num_heads, seq_len, seq_len)
        alibi_bias = -self.slopes.view(-1, 1, 1) * self.distance[:seq_len, :seq_len].unsqueeze(0)
        scores = scores + alibi_bias.unsqueeze(0)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


# =============================================================================
# 5. Gradient Checkpointing
# =============================================================================

class CheckpointedTransformerBlock(nn.Module):
    """
    Transformer Block with Gradient Checkpointing.

    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during forward pass. Instead, they are
    recomputed during backward pass.

    Memory: O(L) → O(sqrt(L)) with checkpointing
    Compute: +33% overhead for recomputation

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        use_checkpoint: Whether to use gradient checkpointing
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.attention = RoPEAttention(d_model, num_heads, max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward computation."""
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional checkpointing."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)


# Simplified RoPE Attention for checkpointing demo
class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate efficiency techniques."""
    print("=" * 80)
    print("EFFICIENCY TECHNIQUES DEMONSTRATION")
    print("=" * 80)

    d_model = 64
    num_heads = 4
    seq_len = 16
    batch_size = 2

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. Flash Attention
    print("\n" + "-" * 80)
    print("1. FLASH ATTENTION (Memory-Efficient)")
    print("-" * 80)

    flash_attn = FlashAttention(d_model, num_heads, block_size=4)
    flash_out = flash_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {flash_out.shape}")
    print("""
    Benefits:
    - O(N) memory instead of O(N²)
    - 2-4x faster on GPU (with CUDA kernel)
    - Enables training on longer sequences
    """)

    # 2. KV Cache
    print("-" * 80)
    print("2. KV CACHE FOR GENERATION")
    print("-" * 80)

    cached_attn = CachedAttention(d_model, num_heads, max_seq_len=128)

    # Simulate generation with cache
    kv_cache = KVCache.create(
        batch_size=1,
        num_layers=1,
        num_heads=num_heads,
        max_seq_len=128,
        d_k=d_model // num_heads,
        device=x.device,
    )

    # First token (prefill)
    first_token = x[:1, :1, :]
    out, kv_cache = cached_attn(first_token, kv_cache, layer_idx=0, use_cache=True)
    print(f"After token 1: cache position = {kv_cache.current_pos}")

    # Subsequent tokens (generation)
    for i in range(2, 5):
        next_token = x[:1, i:i+1, :]
        out, kv_cache = cached_attn(next_token, kv_cache, layer_idx=0, use_cache=True)
        print(f"After token {i}: cache position = {kv_cache.current_pos}")

    # 3. Sliding Window
    print("\n" + "-" * 80)
    print("3. SLIDING WINDOW ATTENTION")
    print("-" * 80)

    sliding_attn = SlidingWindowAttention(d_model, num_heads, window_size=4, max_seq_len=128)
    sliding_out = sliding_attn(x)

    print(f"Window size: 4")
    print(f"Output shape: {sliding_out.shape}")
    print(f"""
    Complexity comparison for seq_len={seq_len}:
    - Full attention: O({seq_len}²) = {seq_len * seq_len} operations
    - Sliding window (W=4): O({seq_len} × 4) = {seq_len * 4} operations
    """)

    # 4. ALiBi
    print("-" * 80)
    print("4. ALiBi (ATTENTION WITH LINEAR BIASES)")
    print("-" * 80)

    alibi_attn = ALiBiAttention(d_model, num_heads, max_seq_len=128)
    alibi_out = alibi_attn(x)

    print(f"ALiBi slopes per head: {alibi_attn.slopes.tolist()}")
    print(f"""
    ALiBi Benefits:
    - No learned positional parameters
    - Better length extrapolation than RoPE
    - Simple to implement (just add bias)
    - Works well for training short, inference long
    """)

    # 5. Gradient Checkpointing
    print("-" * 80)
    print("5. GRADIENT CHECKPOINTING")
    print("-" * 80)

    checkpointed_block = CheckpointedTransformerBlock(
        d_model, d_ff=128, num_heads=num_heads, max_seq_len=128, use_checkpoint=True
    )

    # Compare memory (conceptual)
    print("""
    Memory savings with gradient checkpointing:

    Without checkpointing:
    - Store all L layer activations
    - Memory: O(L × N × D)

    With checkpointing (checkpoint every √L layers):
    - Store √L checkpoints, recompute between
    - Memory: O(√L × N × D)
    - Compute: +33% for recomputation

    Example: 48-layer model
    - Without: 48 × activations
    - With: 7 × activations (85% reduction!)
    """)

    # Summary comparison
    print("\n" + "-" * 80)
    print("EFFICIENCY TECHNIQUES SUMMARY")
    print("-" * 80)
    print("""
    ┌────────────────────────┬─────────────────────┬──────────────────────┐
    │ Technique              │ Memory Benefit      │ Compute Impact       │
    ├────────────────────────┼─────────────────────┼──────────────────────┤
    │ Flash Attention        │ O(N²) → O(N)        │ Faster on GPU        │
    │ KV Cache               │ Recompute → Cache   │ Generation 10x+      │
    │ Sliding Window         │ O(N²) → O(N×W)      │ Proportional to W    │
    │ ALiBi                  │ No pos embed        │ Negligible           │
    │ Gradient Checkpoint    │ O(L) → O(√L)        │ +33% recomputation   │
    │ Mixed Precision        │ FP32 → FP16/BF16    │ 2x faster            │
    └────────────────────────┴─────────────────────┴──────────────────────┘
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Flash Attention: Use for any attention layer (automatic in PyTorch 2.0+)
    2. KV Cache: Essential for fast autoregressive generation
    3. Sliding Window: Great for long documents (Mistral uses this)
    4. ALiBi: Alternative to RoPE with better extrapolation
    5. Gradient Checkpointing: Trade compute for memory in large models

    Next: 12_finetuning.py - LoRA, QLoRA, and Parameter-Efficient Fine-tuning
    """)


if __name__ == "__main__":
    demo()
