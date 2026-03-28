"""
================================================================================
LLM Learning Module 4: ROTARY POSITION EMBEDDINGS (RoPE)
================================================================================

What is RoPE?
-------------
RoPE (Rotary Position Embeddings) encodes position information by rotating
the query and key vectors in the embedding space. Unlike absolute positional
embeddings, RoPE naturally captures relative positions through rotation.

Key Advantages:
1. Relative position awareness (not just absolute)
2. Better length extrapolation
3. No additional parameters to learn
4. Used in LLaMA, GPT-NeoX, PaLM, and many modern LLMs

================================================================================
ILLUSTRATION: How RoPE Works
================================================================================

Concept: Rotate vectors based on their position

    Position 0:  vector ─────────────────────► rotated by angle θ × 0 = 0°
    Position 1:  vector ─────────────────────► rotated by angle θ × 1 = θ
    Position 2:  vector ─────────────────────► rotated by angle θ × 2 = 2θ
    Position 3:  vector ─────────────────────► rotated by angle θ × 3 = 3θ

    2D Rotation Matrix:
        ┌                  ┐
        │ cos(mθ) -sin(mθ) │   where m is position, θ is base angle
        │ sin(mθ)  cos(mθ) │
        └                  ┘

    The rotation angle increases with position, creating a unique encoding.

Why does this capture relative position?
-----------------------------------------
    When we compute Q·K for positions m and n:
        Q_m · K_n = |Q_m| |K_n| cos(angle_m - angle_n)
                  = |Q_m| |K_n| cos((m - n) × θ)

    The dot product depends on the RELATIVE distance (m - n), not absolute positions!

================================================================================
ILLUSTRATION: RoPE Applied to Multi-Dimensional Vectors
================================================================================

For d-dimensional vectors, we:
1. Split into d/2 pairs
2. Rotate each pair with different frequencies

    Vector: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, ...]
            │─────│  │─────│  │─────│  │─────│
            pair 1   pair 2   pair 3   pair 4

    Each pair gets rotated by a different frequency:
        θ₁ = 10000^(-0/d)    (highest frequency)
        θ₂ = 10000^(-2/d)
        θ₃ = 10000^(-4/d)
        ...
        θ_d/2 = 10000^(-(d-2)/d)  (lowest frequency)

    This creates a unique "positional fingerprint" for each position.

================================================================================
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis = cos + i*sin).

    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for the exponential (default 10000.0)

    Returns:
        Complex tensor of shape (max_seq_len, dim/2) containing cos + i*sin

    The frequencies are computed as:
        freq_i = 1 / (base^(2i/d))  for i = 0, 1, ..., d/2-1
    """
    # Compute frequencies: (dim/2,)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices: (max_seq_len,)
    t = torch.arange(max_seq_len)

    # Outer product: (max_seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(iθ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, d_k)
        xk: Key tensor of shape (batch, seq_len, num_heads, d_k)
        freqs_cis: Precomputed frequencies of shape (seq_len, d_k/2)

    Returns:
        Rotated query and key tensors

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Implementation Details:                                                  ║
    ║                                                                           ║
    ║  1. Reshape x from (..., d) to (..., d/2, 2) as complex numbers           ║
    ║  2. Multiply by complex exponential freqs_cis                             ║
    ║  3. Reshape back to (..., d)                                              ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    # Reshape for complex multiplication
    # (batch, seq, heads, d) -> (batch, seq, heads, d/2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex: (batch, seq, heads, d/2)
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # Get the right slice of frequencies
    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    # Reshape freqs for broadcasting: (seq, d/2) -> (1, seq, 1, d/2)
    freqs_cis = freqs_cis.view(1, seq_len, 1, -1)

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Implements the rotation-based positional encoding from:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        dim: Embedding dimension (d_k for each head)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default 10000.0)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  RoPE vs Learned Position Embeddings:                                     ║
    ║                                                                           ║
    ║  Learned PE:                                                              ║
    ║    - Add position vectors to input embeddings                             ║
    ║    - Limited to max_seq_len seen during training                          ║
    ║    - Additional parameters                                                ║
    ║                                                                           ║
    ║  RoPE:                                                                    ║
    ║    - Rotate Q and K vectors before attention                              ║
    ║    - Can extrapolate to longer sequences                                  ║
    ║    - No learned parameters                                                ║
    ║    - Encodes relative positions naturally                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute and cache frequencies
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, base)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            xq: Query tensor (batch, seq_len, num_heads, d_k)
            xk: Key tensor (batch, seq_len, num_heads, d_k)
            start_pos: Starting position (for KV cache during inference)

        Returns:
            Rotated query and key tensors
        """
        seq_len = xq.shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        return apply_rotary_emb(xq, xk, freqs_cis)


class RoPEAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embeddings

    This is the attention mechanism used in LLaMA and other modern LLMs.

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
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.d_k, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Apply RoPE to Q and K (not V!)
        q, k = self.rope(q, k, start_pos)

        # Reshape for attention: (batch, heads, seq, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        causal_mask = self.mask[:, :, start_pos : start_pos + seq_len, : start_pos + seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(attn_out)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate Rotary Position Embeddings.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                          RoPE DEMO                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("ROTARY POSITION EMBEDDINGS (RoPE) DEMONSTRATION")
    print("=" * 80)

    # Hyperparameters
    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads
    max_seq_len = 32
    batch_size = 2
    seq_len = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"d_model: {d_model}, num_heads: {num_heads}, d_k: {d_k}")

    # Precompute frequencies visualization
    print("\n" + "-" * 80)
    print("1. FREQUENCY COMPUTATION")
    print("-" * 80)

    freqs_cis = precompute_freqs_cis(d_k, max_seq_len=16)

    print(f"\nFrequency tensor shape: {freqs_cis.shape}")
    print("(16 positions × 8 frequency components for d_k=16)")

    print("\nFirst 4 positions, first 4 frequencies (as angles in radians):")
    angles = torch.angle(freqs_cis[:4, :4])
    print("Position │  freq_0   freq_1   freq_2   freq_3")
    print("─────────┼─────────────────────────────────────")
    for pos in range(4):
        row = angles[pos].tolist()
        print(f"    {pos}    │ " + "  ".join([f"{a:6.3f}" for a in row]))

    print("""
    Note: Each frequency component has a different period:
    - freq_0: period = 2π (rotates once per position)
    - freq_1: period = 2π × base^(2/d) ≈ slower rotation
    - Higher indices: even slower rotation
    """)

    # RoPE Application
    print("-" * 80)
    print("2. APPLYING RoPE")
    print("-" * 80)

    rope = RotaryEmbedding(d_k, max_seq_len)

    # Create sample Q, K
    q = torch.randn(batch_size, seq_len, num_heads, d_k)
    k = torch.randn(batch_size, seq_len, num_heads, d_k)

    print(f"\nBefore RoPE:")
    print(f"  Q shape: {q.shape}")
    print(f"  Q[0,0,0,:4]: {q[0, 0, 0, :4].tolist()}")

    q_rot, k_rot = rope(q, k)

    print(f"\nAfter RoPE:")
    print(f"  Q shape: {q_rot.shape}")
    print(f"  Q[0,0,0,:4]: {q_rot[0, 0, 0, :4].tolist()}")

    # Relative position property
    print("\n" + "-" * 80)
    print("3. RELATIVE POSITION PROPERTY")
    print("-" * 80)

    # Simulate attention between positions
    print("""
    The key insight: Q·K^T depends on relative distance, not absolute position.

    For positions m and n:
        Q_m · K_n ∝ cos((m - n) × θ)

    Let's verify this with a simple example:
    """)

    # Create simple 2D vectors for visualization
    dim_2d = 2
    rope_2d = RotaryEmbedding(dim_2d, max_seq_len=100)

    # Single vector at different positions
    vec = torch.tensor([[[[1.0, 0.0]]]])  # (1, 1, 1, 2)

    pos_0 = torch.zeros(1, 1, 1, 2)
    pos_3 = torch.zeros(1, 1, 1, 2)

    q_0, _ = rope_2d(vec, pos_0)
    q_3, _ = rope_2d(vec, pos_3)

    print(f"Original vector: {vec[0, 0, 0].tolist()}")
    print(f"Rotated to pos 0: {q_0[0, 0, 0].tolist()}")
    print(f"Rotated to pos 3: {q_3[0, 0, 0].tolist()}")

    # Full attention with RoPE
    print("\n" + "-" * 80)
    print("4. RoPE ATTENTION MODULE")
    print("-" * 80)

    rope_attn = RoPEAttention(d_model, num_heads, max_seq_len)
    out = rope_attn(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in rope_attn.parameters()):,}")

    # Comparison table
    print("\n" + "-" * 80)
    print("5. RoPE vs OTHER POSITION ENCODINGS")
    print("-" * 80)
    print("""
    ┌────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Method             │ Parameters   │ Extrapolation│ Relative Pos     │
    ├────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ Sinusoidal         │ 0            │ Good         │ Implicit         │
    │ Learned Absolute   │ max_len × d  │ Poor         │ No               │
    │ Learned Relative   │ max_len² × d │ Poor         │ Yes              │
    │ RoPE               │ 0            │ Excellent    │ Yes (exact)      │
    │ ALiBi              │ 0            │ Excellent    │ Yes              │
    └────────────────────┴──────────────┴──────────────┴──────────────────┘
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. RoPE rotates Q and K vectors based on position
    2. Rotation angle increases linearly with position
    3. Different dimensions have different rotation frequencies
    4. Q·K^T naturally depends on relative distance (m - n)
    5. No learned parameters - works out of the box
    6. Better length extrapolation than learned embeddings
    7. Used in LLaMA, PaLM, GPT-NeoX, and many modern LLMs

    Next: 05_swiglu.py - SwiGLU Activation Function
    """)


if __name__ == "__main__":
    demo()
