"""
================================================================================
TRANSFORMER BLOCK
================================================================================

What is a Transformer Block?
----------------------------
A transformer block is the basic building unit of a transformer model.
It consists of:
1. Multi-Head Self-Attention (with optional RoPE)
2. Feed-Forward Network (FFN or MoE)
3. Layer Normalization (LayerNorm or RMSNorm)
4. Residual Connections

================================================================================
ILLUSTRATION: Transformer Block Architecture
================================================================================

                    Input
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Pre-Norm Style                               │
    │                                                                      │
    │      ┌─────────────────────────────────────────────────────────┐    │
    │      │                    RMSNorm                               │    │
    │      └─────────────────────────────────────────────────────────┘    │
    │                           │                                          │
    │                           ▼                                          │
    │      ┌─────────────────────────────────────────────────────────┐    │
    │      │              Self-Attention (with RoPE)                  │    │
    │      │                                                          │    │
    │      │   Q ──► RoPE ──┐                                         │    │
    │      │   K ──► RoPE ──┼──► Scaled Dot-Product ──► Output       │    │
    │      │   V ───────────┘                                         │    │
    │      └─────────────────────────────────────────────────────────┘    │
    │                           │                                          │
    │                           ▼                                          │
    │                      [ + ] ◄──── Residual from Input                │
    │                           │                                          │
    │                           ▼                                          │
    │      ┌─────────────────────────────────────────────────────────┐    │
    │      │                    RMSNorm                               │    │
    │      └─────────────────────────────────────────────────────────┘    │
    │                           │                                          │
    │                           ▼                                          │
    │      ┌─────────────────────────────────────────────────────────┐    │
    │      │              SwiGLU FFN (or MoE)                         │    │
    │      │                                                          │    │
    │      │   ──► Linear ──► Swish ──┐                               │    │
    │      │   ──► Linear ────────────┼──► × ──► Linear ──► Output  │    │
    │      └─────────────────────────────────────────────────────────┘    │
    │                           │                                          │
    │                           ▼                                          │
    │                      [ + ] ◄──── Residual                           │
    │                           │                                          │
    └───────────────────────────┼──────────────────────────────────────────┘
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

# Import components from previous modules (inline for standalone use)
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 1376  # ~2.67 * d_model for SwiGLU
    max_seq_len: int = 2048
    dropout: float = 0.1
    num_experts: int = 8  # For MoE, 0 = dense FFN
    top_k: int = 2  # For MoE routing


# =============================================================================
# Core Components (repeated for standalone module)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class RoPEAttention(nn.Module):
    """Multi-Head Attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        freqs_cis = precompute_freqs_cis(self.d_k, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def _apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_r = x.float().reshape(*x.shape[:-1], -1, 2)
        x_c = torch.view_as_complex(x_r)
        freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
        return torch.view_as_real(x_c * freqs_cis).flatten(-2).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        q = self._apply_rope(q, self.freqs_cis[:seq_len])
        k = self._apply_rope(k, self.freqs_cis[:seq_len])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Expert(nn.Module):
    """Single Expert for MoE."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x))))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        scores = self.router(x_flat)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            token_indices, k_indices = torch.where(top_k_indices == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            weights = top_k_weights[token_indices, k_indices]
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, d_model)


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer Block with Pre-Norm, RoPE, and SwiGLU/MoE.

    This is the standard block used in modern LLMs like LLaMA.

    Args:
        config: TransformerConfig object

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Pre-Norm vs Post-Norm:                                                   ║
    ║                                                                           ║
    ║  Post-Norm (original transformer):                                        ║
    ║    x = norm(x + attention(x))                                             ║
    ║    x = norm(x + ffn(x))                                                   ║
    ║                                                                           ║
    ║  Pre-Norm (modern LLMs like LLaMA):                                       ║
    ║    x = x + attention(norm(x))                                             ║
    ║    x = x + ffn(norm(x))                                                   ║
    ║                                                                           ║
    ║  Pre-Norm benefits:                                                       ║
    ║  - More stable training (gradients flow directly through residuals)       ║
    ║  - No need for learning rate warmup                                       ║
    ║  - Works better for very deep networks                                    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Attention
        self.attention = RoPEAttention(
            config.d_model,
            config.num_heads,
            config.max_seq_len,
            config.dropout,
        )

        # FFN (dense or MoE)
        if config.num_experts > 0:
            self.ffn = MoEFFN(
                config.d_model,
                config.d_ff,
                config.num_experts,
                config.top_k,
                config.dropout,
            )
            self.is_moe = True
        else:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)
            self.is_moe = False

        # Normalization
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate the Transformer Block.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                     TRANSFORMER BLOCK DEMO                                ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("TRANSFORMER BLOCK DEMONSTRATION")
    print("=" * 80)

    # Configuration for a small model
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=1,
        d_ff=683,  # ~2.67 * 256
        max_seq_len=512,
        dropout=0.1,
        num_experts=0,  # Dense FFN
    )

    batch_size = 2
    seq_len = 16

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Config: d_model={config.d_model}, num_heads={config.num_heads}")
    print(f"        d_ff={config.d_ff}, max_seq_len={config.max_seq_len}")

    # Dense block
    print("\n" + "-" * 80)
    print("1. DENSE TRANSFORMER BLOCK")
    print("-" * 80)

    dense_block = TransformerBlock(config)
    dense_out = dense_block(x)

    print(f"\nOutput shape: {dense_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in dense_block.parameters()):,}")

    # Break down by component
    attn_params = sum(p.numel() for p in dense_block.attention.parameters())
    ffn_params = sum(p.numel() for p in dense_block.ffn.parameters())
    norm_params = sum(p.numel() for p in dense_block.attention_norm.parameters())
    norm_params += sum(p.numel() for p in dense_block.ffn_norm.parameters())

    print(f"\nParameter breakdown:")
    print(f"  Attention:   {attn_params:,} ({attn_params / sum(p.numel() for p in dense_block.parameters()) * 100:.1f}%)")
    print(f"  FFN:         {ffn_params:,} ({ffn_params / sum(p.numel() for p in dense_block.parameters()) * 100:.1f}%)")
    print(f"  Norms:       {norm_params:,} ({norm_params / sum(p.numel() for p in dense_block.parameters()) * 100:.1f}%)")

    # MoE block
    print("\n" + "-" * 80)
    print("2. MOE TRANSFORMER BLOCK")
    print("-" * 80)

    moe_config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=1,
        d_ff=683,
        max_seq_len=512,
        dropout=0.1,
        num_experts=8,
        top_k=2,
    )

    moe_block = TransformerBlock(moe_config)
    moe_out = moe_block(x)

    print(f"\nOutput shape: {moe_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in moe_block.parameters()):,}")

    moe_ffn_params = sum(p.numel() for p in moe_block.ffn.parameters())
    active_per_token = (config.d_model * config.d_ff + config.d_ff * config.d_model) * moe_config.top_k
    print(f"FFN parameters: {moe_ffn_params:,}")
    print(f"Active FFN params per token: ~{active_per_token:,} ({moe_config.top_k}/{moe_config.num_experts} experts)")

    # Gradient flow visualization
    print("\n" + "-" * 80)
    print("3. GRADIENT FLOW THROUGH BLOCK")
    print("-" * 80)
    print("""
    Pre-norm allows gradients to flow directly through residual connections:

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │    x ─────────────────────────────────────────────────┐                │
    │    │                                                   │                │
    │    ▼                                                   │                │
    │  [RMSNorm]                                             │                │
    │    │                                                   │                │
    │    ▼                                                   │                │
    │  [Attention]                                           │                │
    │    │                                                   │                │
    │    ▼                                                   ▼                │
    │  [Dropout] ────────────────────────────────────────► [+]               │
    │                                                        │                │
    │                                                        │                │
    │    x + attention(norm(x)) ─────────────────────────────┘                │
    │    │                                                                   │
    │    │         ┌────────────────────────────────────────────────┐        │
    │    │         │ Same pattern for FFN                            │        │
    │    │         │ x + ffn(norm(x))                                │        │
    │    │         └────────────────────────────────────────────────┘        │
    │    ▼                                                                   │
    │  Output                                                                │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    Key: The gradient can skip the attention/FFN entirely via the residual path!
    """)

    # Memory and compute comparison
    print("\n" + "-" * 80)
    print("4. COMPUTE & MEMORY COMPARISON (per block, d_model=768)")
    print("-" * 80)

    d_model = 768
    d_ff = int(8 * d_model / 3)
    num_heads = 12

    # Attention: 4 matrices of d×d
    attn_params = 4 * d_model * d_model
    # FFN: 3 matrices of d×d_ff (SwiGLU)
    ffn_params = 3 * d_model * d_ff

    print(f"\nDense block:")
    print(f"  Attention:  {attn_params / 1e6:.2f}M params")
    print(f"  FFN:        {ffn_params / 1e6:.2f}M params")
    print(f"  Total:      {(attn_params + ffn_params) / 1e6:.2f}M params")

    # MoE with 8 experts
    num_experts = 8
    moe_ffn_params = num_experts * 3 * d_model * d_ff
    print(f"\nMoE block (8 experts):")
    print(f"  Attention:  {attn_params / 1e6:.2f}M params")
    print(f"  FFN:        {moe_ffn_params / 1e6:.2f}M params")
    print(f"  Total:      {(attn_params + moe_ffn_params) / 1e6:.2f}M params")
    print(f"  Active:     {(attn_params + ffn_params * 2) / 1e6:.2f}M params (top-2)")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Transformer block = Attention + FFN with residual connections
    2. Pre-norm is more stable than post-norm for deep networks
    3. RoPE encodes position in attention (no learned positional embeddings)
    4. SwiGLU provides smooth gating for FFN
    5. MoE can scale parameters while keeping compute constant
    6. Residual connections allow gradients to skip layers

    Next: architecture/mixture_of_experts.py - Sparse Architecture (MoE)
    """)


if __name__ == "__main__":
    demo()
