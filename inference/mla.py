"""
================================================================================
MULTI-HEAD LATENT ATTENTION (MLA)
================================================================================

DeepSeek-V2/V3's breakthrough attention mechanism that drastically reduces
KV Cache memory during inference.

Standard Multi-Head Attention:
    KV Cache stores: (batch, num_layers, num_heads, seq_len, d_k)
    For d_model=7168, num_heads=128, d_k=56, seq_len=32K:
    Per layer KV Cache = 2 × 128 × 32768 × 56 × 2 bytes ≈ 939 MB

Multi-Head Latent Attention (MLA):
    KV Cache stores: (batch, num_layers, seq_len, d_c)
    Where d_c << d_model (e.g., d_c = 512)
    Per layer KV Cache = 2 × 32768 × 512 × 2 bytes ≈ 67 MB
    
    Compression ratio: ~14x!

================================================================================
ILLUSTRATION: How MLA Works
================================================================================

Standard Attention (at inference time):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Input h_t ──► W_k ──► K_t  ──┐                                     │
    │              W_v ──► V_t  ────┼──► Attention ──► Output              │
    │              W_q ──► Q_t  ────┘                                     │
    │                                                                      │
    │  Cache: [K_1, K_2, ..., K_t] and [V_1, V_2, ..., V_t]              │
    │  Memory: O(seq_len × num_heads × d_k)                              │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

MLA (at inference time):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Input h_t ──► W_CKV ──► c_t^KV  ──► Cache                          │
    │                                                                      │
    │  During Attention:                                                   │
    │    c_t^KV ──► W_DK ──► K_t  ──┐                                     │
    │    c_t^KV ──► W_DV ──► V_t  ──┼──► Attention ──► Output              │
    │    h_t ──► W_CQ ──► c_t^Q ──► W_DQ ──► Q_t ────┘                   │
    │                                                                      │
    │  Cache: [c_1^KV, c_2^KV, ..., c_t^KV]  (single tensor!)            │
    │  Memory: O(seq_len × d_c)  where d_c << num_heads × d_k            │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Key Innovation:
    Instead of caching K and V separately, cache their shared LOW-RANK
    representation c^KV. During inference, decompress c^KV into K and V.
    Since d_c is much smaller than the full KV dimension, this saves
    enormous amounts of memory.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - Simplified Implementation

    As described in DeepSeek-V2: "A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model"

    Key idea: Compress keys and values into a shared latent vector c^KV
    during inference. Only cache c^KV, not full K and V.

    For simplicity, this implementation uses a standard low-rank factorization
    approach. The full DeepSeek implementation also includes decoupled RoPE
    (where position info is handled separately from content).

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_c: Latent compression dimension (much smaller than d_model)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Parameter Comparison (d_model=2048, num_heads=16, d_k=128):            ║
    ║                                                                           ║
    ║  Standard MHA:                                                            ║
    ║    W_q, W_k, W_v: 3 × (2048 × 2048) = 12.6M                             ║
    ║    KV Cache per token: 2 × 16 × 128 = 4,096 values                      ║
    ║                                                                           ║
    ║  MLA (d_c = 512):                                                         ║
    ║    W_cq, W_ckv: 2 × (2048 × 512) = 2.1M                                 ║
    ║    W_dq, W_dk, W_dv: 3 × (512 × 2048) = 3.1M                            ║
    ║    Total params: ~5.2M (vs 12.6M for MHA)                               ║
    ║    KV Cache per token: 2 × 512 = 1,024 values                           ║
    ║    Cache reduction: 4x!                                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_c: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_c = d_c

        # Q compression and decompression
        self.w_cq = nn.Linear(d_model, d_c, bias=False)
        self.w_dq = nn.Linear(d_c, d_model, bias=False)

        # KV compression and decompression (shared!)
        self.w_ckv = nn.Linear(d_model, d_c, bias=False)
        self.w_dk = nn.Linear(d_c, d_model, bias=False)
        self.w_dv = nn.Linear(d_c, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        return_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional KV cache.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            kv_cache: Optional cached c^KV from previous steps
                      Shape: (batch, prev_seq_len, d_c)
            return_kv_cache: Whether to return updated cache

        Returns:
            output: Attention output (batch, seq_len, d_model)
            kv_cache: Updated c^KV cache if return_kv_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Compress Q
        c_q = self.w_cq(x)  # (batch, seq, d_c)
        q = self.w_dq(c_q)  # (batch, seq, d_model)

        # Compress KV
        c_kv_new = self.w_ckv(x)  # (batch, seq, d_c)

        # Merge with cache if provided
        if kv_cache is not None:
            c_kv = torch.cat([kv_cache, c_kv_new], dim=1)
        else:
            c_kv = c_kv_new

        # Decompress K and V from shared latent representation
        k = self.w_dk(c_kv)  # (batch, total_seq, d_model)
        v = self.w_dv(c_kv)  # (batch, total_seq, d_model)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        total_seq_len = k.shape[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask
        causal_mask = self.mask[:, :, total_seq_len - seq_len : total_seq_len, :total_seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.w_o(attn_out)

        if return_kv_cache:
            return output, c_kv
        return output, None


class StandardMHA(nn.Module):
    """Standard Multi-Head Attention for comparison."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv_cache: bool = False,
    ):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)
        k_new = self.w_k(x)
        v_new = self.w_v(x)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k_new], dim=1)
            v = torch.cat([kv_cache[1], v_new], dim=1)
        else:
            k, v = k_new, v_new

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        total_seq = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = self.mask[:, :, total_seq - seq_len : total_seq, :total_seq]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_out)

        if return_kv_cache:
            return output, (k, v)
        return output, None


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate MLA and compare with standard MHA."""
    print("=" * 80)
    print("MULTI-HEAD LATENT ATTENTION (MLA)")
    print("=" * 80)

    d_model = 2048
    num_heads = 16
    d_c = 512
    batch_size = 2
    seq_len = 32

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k: {d_model // num_heads}")
    print(f"  MLA latent dim d_c: {d_c}")
    print(f"  Input shape: {x.shape}")

    # Standard MHA
    print("\n" + "-" * 80)
    print("1. STANDARD MULTI-HEAD ATTENTION")
    print("-" * 80)

    mha = StandardMHA(d_model, num_heads)
    mha_out, mha_cache = mha(x, return_kv_cache=True)

    mha_params = sum(p.numel() for p in mha.parameters())
    k_cache_size = mha_cache[0].numel()
    v_cache_size = mha_cache[1].numel()
    mha_cache_total = k_cache_size + v_cache_size

    print(f"\nParameters: {mha_params:,}")
    print(f"Output shape: {mha_out.shape}")
    print(f"KV Cache per batch: K={k_cache_size:,} + V={v_cache_size:,} = {mha_cache_total:,} values")

    # MLA
    print("\n" + "-" * 80)
    print("2. MULTI-HEAD LATENT ATTENTION (MLA)")
    print("-" * 80)

    mla = MultiHeadLatentAttention(d_model, num_heads, d_c)
    mla_out, mla_cache = mla(x, return_kv_cache=True)

    mla_params = sum(p.numel() for p in mla.parameters())
    mla_cache_total = mla_cache.numel()

    print(f"\nParameters: {mla_params:,}")
    print(f"Output shape: {mla_out.shape}")
    print(f"Latent KV Cache per batch: {mla_cache_total:,} values")

    # Comparison
    print("\n" + "-" * 80)
    print("3. COMPARISON")
    print("-" * 80)

    cache_reduction = mha_cache_total / mla_cache_total
    param_ratio = mla_params / mha_params

    print(f"""
    ┌────────────────────┬──────────────────┬──────────────────┐
    │ Metric             │ Standard MHA     │ MLA (d_c={d_c})   │
    ├────────────────────┼──────────────────┼──────────────────┤
    │ Parameters         │ {mha_params:>12,}   │ {mla_params:>12,}   │
    │ KV Cache (values)  │ {mha_cache_total:>12,}   │ {mla_cache_total:>12,}   │
    │ Cache Reduction    │       1.0×       │       {cache_reduction:.1f}×       │
    │ Param Ratio        │       1.0×       │       {param_ratio:.2f}×       │
    └────────────────────┴──────────────────┴──────────────────┘

    At inference with seq_len=32K, batch=1:
    Standard MHA KV Cache: ~{(mha_cache_total * 32000 / seq_len * 2 / 1024 / 1024):.1f} MB
    MLA KV Cache:         ~{(mla_cache_total * 32000 / seq_len * 2 / 1024 / 1024):.1f} MB
    """)

    # Autoregressive generation simulation
    print("-" * 80)
    print("4. AUTOREGRESSIVE GENERATION WITH KV CACHE")
    print("-" * 80)

    # Prefill
    prefill_x = x[:, :8, :]  # First 8 tokens
    _, mla_kv = mla(prefill_x, return_kv_cache=True)

    print(f"\nPrefill (8 tokens):")
    print(f"  MLA cache shape: {mla_kv.shape}")

    # Generate 4 more tokens
    for i in range(4):
        next_token = x[:, 8 + i : 9 + i, :]
        _, mla_kv = mla(next_token, kv_cache=mla_kv, return_kv_cache=True)
        print(f"  After token {i+1}: cache shape = {mla_kv.shape}")

    print("""
    Key observation:
    - Standard MHA cache grows by (num_heads × d_k) per token
    - MLA cache grows by only d_c per token
    - For d_model=7168, num_heads=128, d_k=56: standard = 7168 values/token
    - MLA with d_c=512: only 512 values/token → 14x reduction!
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. MLA compresses K and V into a shared latent vector c^KV
    2. Only c^KV is cached during inference, not full K and V
    3. Cache reduction is (num_heads × d_k) / d_c, often 10-20x
    4. Parameters are also reduced vs standard MHA
    5. DeepSeek-V2/V3 use MLA to support 128K+ context efficiently
    6. Trade-off: slight quality loss possible if d_c too small

    Next: architecture/hyper_connections.py - Manifold-Constrained Hyper-Connections
    """)


if __name__ == "__main__":
    demo()
