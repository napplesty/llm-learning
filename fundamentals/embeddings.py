"""
================================================================================
EMBEDDINGS
================================================================================

What are Embeddings?
--------------------
Embeddings convert discrete tokens (integers) into continuous dense vectors.
This allows the model to learn semantic relationships between tokens.

Key Concepts:
1. Token Embeddings: Map each token ID to a learnable vector
2. Position Embeddings: Add positional information (absolute or relative)
3. Layer Normalization: Stabilize training by normalizing activations

================================================================================
ILLUSTRATION: From Tokens to Embeddings
================================================================================

Input Text: "hello world"
    ↓ Tokenizer
Token IDs: [1042, 3891]
    ↓ Token Embedding Layer

    ┌─────────────────────────────────────────────────────────────┐
    │                    Embedding Matrix                         │
    │                    (vocab_size × d_model)                   │
    │                                                             │
    │     ID 0   →  [0.1, -0.2, 0.3, ...]                        │
    │     ID 1   →  [0.5, 0.1, -0.4, ...]                        │
    │     ...                                                     │
    │     ID 1042 →  [0.8, 0.3, -0.1, ...]  ← "hello"            │
    │     ...                                                     │
    │     ID 3891 →  [-0.2, 0.9, 0.4, ...]  ← "world"            │
    └─────────────────────────────────────────────────────────────┘
    ↓ Lookup

Token Embeddings:
    "hello" → [0.8, 0.3, -0.1, ...]  (shape: d_model)
    "world" → [-0.2, 0.9, 0.4, ...]  (shape: d_model)

    Combined: shape (seq_len, d_model) = (2, d_model)

    ↓ + Position Embeddings

Position-Aware Embeddings:
    Position 0: [0.8, 0.3, -0.1, ...] + [0.1, 0.0, 0.2, ...]
    Position 1: [-0.2, 0.9, 0.4, ...] + [-0.1, 0.1, 0.0, ...]

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer

    Converts token IDs to dense vectors using a learnable embedding matrix.

    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of embedding vectors

    Shape:
        Input: (batch_size, seq_len) - token IDs
        Output: (batch_size, seq_len, d_model) - embedded vectors

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Embedding Matrix Shape: (vocab_size, d_model)                            ║
    ║  For GPT-2 small: vocab_size=50,257, d_model=768                          ║
    ║  Parameters: 50,257 × 768 = 38,597,376 (~39M params)                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Initialize with scaled normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding (from "Attention Is All You Need")

    Uses fixed (non-learned) sinusoidal functions to encode position.
    This allows the model to extrapolate to longer sequences.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length
        d_model: Dimension of embedding vectors

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Why Sinusoidal?                                                          ║
    ║  - PE(pos + k) can be expressed as linear function of PE(pos)             ║
    ║  - Allows model to learn relative positions                               ║
    ║  - Can extrapolate to longer sequences than seen during training          ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the divisor term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positional encoding of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding

    Instead of fixed sinusoidal encoding, learn position embeddings.
    Used in GPT-2, BERT, and many modern models.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Dimension of embedding vectors
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positional embedding of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    A simplified version of LayerNorm without mean centering.
    More efficient and works well for LLMs.

    Formula: x * rsqrt(mean(x^2) + eps) * weight

    Args:
        d_model: Dimension of the input
        eps: Small constant for numerical stability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  LayerNorm vs RMSNorm:                                                    ║
    ║                                                                           ║
    ║  LayerNorm: x_norm = (x - mean) / std * gamma + beta                      ║
    ║  RMSNorm:   x_norm = x / rms * gamma     (no mean, no beta)               ║
    ║                                                                           ║
    ║  RMSNorm is used in LLaMA, Gopher, and other modern LLMs                  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization

    Normalizes across the feature dimension, stabilizing training.
    Used in original Transformer and BERT.

    Args:
        d_model: Dimension of the input
        eps: Small constant for numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps) * self.weight + self.bias


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate embedding layers with visualizations.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         EMBEDDINGS DEMO                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("EMBEDDINGS DEMONSTRATION")
    print("=" * 80)

    # Hyperparameters (small model)
    vocab_size = 1000
    d_model = 64
    max_seq_len = 128
    batch_size = 2
    seq_len = 10

    # Create sample input
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\nInput token IDs shape: {token_ids.shape}")
    print(f"Sample tokens:\n{token_ids}")

    # Token Embeddings
    print("\n" + "-" * 80)
    print("1. TOKEN EMBEDDINGS")
    print("-" * 80)

    token_emb = TokenEmbedding(vocab_size, d_model)
    token_out = token_emb(token_ids)
    print(f"Token embedding output shape: {token_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in token_emb.parameters()):,}")

    # Sinusoidal Positional Embeddings
    print("\n" + "-" * 80)
    print("2. SINUSOIDAL POSITIONAL EMBEDDINGS")
    print("-" * 80)

    pos_emb = PositionalEmbedding(max_seq_len, d_model)
    pos_out = pos_emb(token_out)
    print(f"Positional encoding shape: {pos_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in pos_emb.parameters()):,} (fixed, no learning)")

    # Learned Positional Embeddings
    print("\n" + "-" * 80)
    print("3. LEARNED POSITIONAL EMBEDDINGS")
    print("-" * 80)

    learned_pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)
    learned_pos_out = learned_pos_emb(token_out)
    print(f"Learned positional embedding shape: {learned_pos_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in learned_pos_emb.parameters()):,}")

    # Combined Embeddings
    print("\n" + "-" * 80)
    print("4. COMBINED EMBEDDINGS (Token + Position)")
    print("-" * 80)

    combined = token_out + learned_pos_out
    print(f"Combined embedding shape: {combined.shape}")

    # Normalization
    print("\n" + "-" * 80)
    print("5. NORMALIZATION COMPARISON")
    print("-" * 80)

    ln = LayerNorm(d_model)
    rms = RMSNorm(d_model)

    ln_out = ln(combined)
    rms_out = rms(combined)

    print(f"LayerNorm output mean: {ln_out.mean(dim=-1)[0, :3].tolist()}")
    print(f"LayerNorm output std:  {ln_out.std(dim=-1)[0, :3].tolist()}")
    print(f"RMSNorm output mean:   {rms_out.mean(dim=-1)[0, :3].tolist()}")
    print(f"RMSNorm output std:    {rms_out.std(dim=-1)[0, :3].tolist()}")

    print("\n" + "-" * 80)
    print("6. PARAMETER COUNT COMPARISON")
    print("-" * 80)

    print(f"\nLayerNorm params: {sum(p.numel() for p in ln.parameters()):,} (weight + bias)")
    print(f"RMSNorm params:   {sum(p.numel() for p in rms.parameters()):,} (weight only)")

    # Visualize sinusoidal pattern
    print("\n" + "-" * 80)
    print("7. SINUSOIDAL POSITIONAL ENCODING PATTERN")
    print("-" * 80)
    print("""
    Position Encoding Visualization (first 4 positions, first 8 dims):

    Position  Dim0    Dim1    Dim2    Dim3    Dim4    Dim5    Dim6    Dim7
    ─────────────────────────────────────────────────────────────────────────
        0    │ 0.00   1.00    0.00    1.00    0.00    1.00    0.00    1.00
        1    │ 0.84   0.54    0.01    1.00    0.00    1.00    0.00    1.00
        2    │ 0.91  -0.42    0.02    1.00    0.00    1.00    0.00    1.00
        3    │ 0.14  -0.99    0.03    1.00    0.00    1.00    0.00    1.00

    Lower dimensions vary faster (sin/cos of higher frequency)
    Higher dimensions vary slower (sin/cos of lower frequency)
    This creates a unique "signature" for each position.
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Token embeddings convert discrete IDs to continuous vectors
    2. Positional embeddings add order information (crucial for attention)
    3. Sinusoidal: fixed, can extrapolate; Learned: flexible, limited to max_len
    4. RMSNorm is simpler and faster than LayerNorm (no mean centering)
    5. Scale factor sqrt(d_model) helps with gradient flow

    Next: fundamentals/attention.py - Attention Mechanisms
    """)


if __name__ == "__main__":
    demo()
