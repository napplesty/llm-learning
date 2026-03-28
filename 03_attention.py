"""
================================================================================
LLM Learning Module 3: ATTENTION
================================================================================

What is Attention?
------------------
Attention allows the model to focus on different parts of the input when
producing each part of the output. It's the core mechanism of transformers.

Key Concepts:
1. Query (Q): What I'm looking for
2. Key (K): What I can offer
3. Value (V): What I actually provide
4. Attention Score: How much Q matches K

================================================================================
ILLUSTRATION: Scaled Dot-Product Attention
================================================================================

                    Query (Q)
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │    Q × K^T    ──►    Scale    ──►    Mask    ──►    Softmax     │
    │                      (√d_k)           (opt)                     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
                  Attention Weights
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │                    Attention Weights × V                         │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
                   Output

    Formula: Attention(Q, K, V) = softmax(Q K^T / √d_k) V

    Shapes:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_v)
        Output: (batch, heads, seq_len, d_v)

================================================================================
ILLUSTRATION: Multi-Head Attention
================================================================================

                    Input
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Split into multiple heads (each learns different patterns)      │
    │                                                                  │
    │    Head 1    Head 2    Head 3    Head 4    ...    Head h         │
    │    (Q₁K₁V₁)  (Q₂K₂V₂)  (Q₃K₃V₃)  (Q₄K₄V₄)        (QₕKₕVₕ)        │
    │       │         │         │         │              │             │
    │       ▼         ▼         ▼         ▼              ▼             │
    │    Attn₁     Attn₂     Attn₃     Attn₄    ...   Attnₕ           │
    │       │         │         │         │              │             │
    └───────┴─────────┴─────────┴─────────┴──────────────┴─────────────┘
                       │
                       ▼
                    Concat
                       │
                       ▼
                   Linear Layer
                       │
                       ▼
                    Output

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, d_k)
        k: Key tensor of shape (batch, heads, seq_len, d_k)
        v: Value tensor of shape (batch, heads, seq_len, d_v)
        mask: Optional mask of shape (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        dropout: Optional dropout layer

    Returns:
        output: Attention output of shape (batch, heads, seq_len, d_v)
        attn_weights: Attention weights of shape (batch, heads, seq_len, seq_len)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Why scale by √d_k?                                                       ║
    ║                                                                           ║
    ║  For large d_k, dot products grow large → softmax becomes very peaked     ║
    ║  → gradients become very small (vanishing gradients)                      ║
    ║                                                                           ║
    ║  Scaling counteracts this by keeping values in a reasonable range         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    d_k = q.size(-1)

    # Compute attention scores: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (for causal attention, mask future positions)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Compute output: (batch, heads, seq_len, d_v)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Why Multi-Head?                                                          ║
    ║                                                                           ║
    ║  Single head: all positions compete for one attention distribution        ║
    ║  Multi-head: each head can focus on different aspects:                    ║
    ║    - Head 1: syntactic relationships                                      ║
    ║    - Head 2: semantic relationships                                       ║
    ║    - Head 3: positional patterns                                          ║
    ║    - ...                                                                  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor of shape (batch, seq_len, d_model)
            k: Key tensor of shape (batch, seq_len, d_model)
            v: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size = q.size(0)

        # 1. Linear projections
        # Shape: (batch, seq_len, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2. Reshape for multi-head attention
        # Shape: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout
        )

        # 4. Concatenate heads
        # Shape: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. Final linear projection
        output = self.w_o(attn_out)

        return output


class CausalSelfAttention(nn.Module):
    """
    Causal (Autoregressive) Self-Attention

    Used in decoder-only models like GPT. Each position can only attend
    to previous positions (no peeking into the future).

    The causal mask looks like:
        Position:  0  1  2  3
        ─────────────────────
             0  │  1  0  0  0
             1  │  1  1  0  0
             2  │  1  1  1  0
             3  │  1  1  1  1

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        # Create causal mask (lower triangular)
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Apply causal mask (only allow attending to past)
        causal_mask = self.mask[:, :, :seq_len, :seq_len]

        # Self-attention: Q, K, V all come from same input
        attn_out = self.mha(x, x, x, causal_mask)

        return self.dropout(attn_out)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    Used in LLaMA 2/3 and other modern LLMs. Reduces memory and computation
    by sharing Key and Value heads among multiple Query heads.

    Standard Multi-Head Attention: each Q head has its own K, V head
    Grouped Query Attention: multiple Q heads share K, V heads
    Multi-Query Attention (extreme): all Q heads share single K, V head

    Example with 8 Q heads and 2 KV groups:
        Q: [h0, h1, h2, h3, h4, h5, h6, h7]  (8 heads)
        K: [g0, g0, g0, g0, g1, g1, g1, g1]  (2 groups)
        V: [g0, g0, g0, g0, g1, g1, g1, g1]  (2 groups)

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  GQA Benefits:                                                            ║
    ║  - Reduced KV cache size during inference (important for long sequences)  ║
    ║  - Better quality than multi-query attention                              ║
    ║  - Faster inference than standard multi-head attention                    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # Q heads per KV group
        self.d_k = d_model // num_heads

        # Q projection (full heads)
        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # K, V projections (reduced heads)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        # Output projection
        self.w_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.w_q(x)  # (batch, seq, num_heads * d_k)
        k = self.w_k(x)  # (batch, seq, num_kv_heads * d_k)
        v = self.w_v(x)  # (batch, seq, num_kv_heads * d_k)

        # Reshape Q: (batch, num_heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Reshape K, V: (batch, num_kv_heads, seq, d_k)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Expand K, V to match Q heads (repeat each KV head for its group)
        # (batch, num_kv_heads, seq, d_k) -> (batch, num_heads, seq, d_k)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        d_k = self.d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply causal mask
        causal_mask = self.mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_k
        )

        return self.w_o(attn_out)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate attention mechanisms.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        ATTENTION DEMO                                     ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("ATTENTION MECHANISMS DEMONSTRATION")
    print("=" * 80)

    # Hyperparameters
    d_model = 64
    num_heads = 4
    max_seq_len = 32
    batch_size = 2
    seq_len = 8

    # Create sample input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"d_model: {d_model}, num_heads: {num_heads}, d_k per head: {d_model // num_heads}")

    # Multi-Head Attention
    print("\n" + "-" * 80)
    print("1. MULTI-HEAD ATTENTION")
    print("-" * 80)

    mha = MultiHeadAttention(d_model, num_heads)
    mha_out = mha(x, x, x)  # Self-attention
    print(f"Output shape: {mha_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in mha.parameters()):,}")

    # Causal Self-Attention
    print("\n" + "-" * 80)
    print("2. CAUSAL SELF-ATTENTION")
    print("-" * 80)

    causal_attn = CausalSelfAttention(d_model, num_heads, max_seq_len)
    causal_out = causal_attn(x)
    print(f"Output shape: {causal_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in causal_attn.parameters()):,}")

    # Visualize causal mask
    print("\nCausal mask visualization (8x8):")
    mask = torch.tril(torch.ones(8, 8)).int()
    for row in mask:
        print("  " + " ".join(["█" if m else "·" for m in row.tolist()]))

    # Grouped Query Attention
    print("\n" + "-" * 80)
    print("3. GROUPED QUERY ATTENTION")
    print("-" * 80)

    gqa = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=2, max_seq_len=max_seq_len)
    gqa_out = gqa(x)
    print(f"Output shape: {gqa_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in gqa.parameters()):,}")
    print(f"Query heads: 8, KV heads: 2, Group size: {8 // 2}")

    # Compare parameter counts
    print("\n" + "-" * 80)
    print("4. PARAMETER COMPARISON")
    print("-" * 80)

    # Standard MHA with 8 heads
    mha_8 = MultiHeadAttention(d_model, num_heads=8)
    # GQA with 8 Q heads, 2 KV heads
    gqa_8_2 = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=2, max_seq_len=max_seq_len)
    # GQA with 8 Q heads, 1 KV head (Multi-Query Attention)
    gqa_8_1 = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=1, max_seq_len=max_seq_len)

    print(f"\nStandard MHA (8 heads):    {sum(p.numel() for p in mha_8.parameters()):,} params")
    print(f"GQA (8 Q, 2 KV heads):     {sum(p.numel() for p in gqa_8_2.parameters()):,} params")
    print(f"MQA (8 Q, 1 KV head):      {sum(p.numel() for p in gqa_8_1.parameters()):,} params")

    savings = 1 - sum(p.numel() for p in gqa_8_2.parameters()) / sum(p.numel() for p in mha_8.parameters())
    print(f"\nGQA saves {savings*100:.1f}% parameters compared to standard MHA")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Attention computes weighted sum of values, weights based on query-key similarity
    2. Multi-head attention allows learning different relationship patterns
    3. Causal masking prevents attending to future positions (for autoregressive models)
    4. Grouped Query Attention (GQA) reduces KV cache size while maintaining quality
    5. Scale factor √d_k prevents softmax from becoming too peaked

    Next: 04_rope.py - Rotary Position Embeddings
    """)


if __name__ == "__main__":
    demo()
