"""
================================================================================
LLM Learning Module 8: COMPLETE LLM MODEL
================================================================================

This module assembles all the components into a complete language model:
- Token Embeddings
- Multiple Transformer Blocks
- Output Head for Language Modeling

Target: ~0.1B (100M) parameter model

================================================================================
ILLUSTRATION: Complete LLM Architecture
================================================================================

    Input Tokens: [T1, T2, T3, T4, T5]
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Token Embedding                                  │
    │                    (vocab_size × d_model)                               │
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Transformer Block 1                                 │
    │  ┌─────────────────────────────────────────────────────────────────────┐│
    │  │ RMSNorm → RoPE Attention → Add                                     ││
    │  │ RMSNorm → SwiGLU/MoE FFN  → Add                                     ││
    │  └─────────────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Transformer Block 2                                 │
    │                          ... same as above                              │
    └─────────────────────────────────────────────────────────────────────────┘
         │
        ...
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Transformer Block N                                 │
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Final RMSNorm                                    │
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       Output Linear                                      │
    │                    (d_model → vocab_size)                               │
    │                                                                          │
    │    Note: Often uses tied embeddings (same as input embeddings)          │
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Softmax                                           │
    │            (or LogSoftmax for numerical stability)                      │
    └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    Output Probabilities: [P(T1), P(T2), P(T3), ..., P(vocab_size)]
    per position

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Import from previous modules
from importlib.util import spec_from_file_location, module_from_spec
import os


@dataclass
class LLMConfig:
    """
    Configuration for the complete LLM.

    This configuration targets ~0.1B (100M) parameters.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Parameter Count Estimation:                                              ║
    ║                                                                           ║
    ║  Total ≈ vocab_size × d_model +                                          ║
    ║          num_layers × (4 × d_model² + 3 × d_model × d_ff) +              ║
    ║          d_model × vocab_size (output)                                   ║
    ║                                                                           ║
    ║  For our config (approx):                                                 ║
    ║    Embeddings: 10,000 × 256 = 2.56M                                       ║
    ║    Per layer:  4×256² + 3×256×683 ≈ 0.68M                                ║
    ║    12 layers:  0.68M × 12 = 8.16M                                         ║
    ║    Output:     256 × 10,000 = 2.56M                                       ║
    ║    Total:      ≈ 13.3M (without MoE)                                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    vocab_size: int = 10000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 683  # ~2.67 × d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    num_experts: int = 0  # 0 = dense FFN, >0 = MoE
    top_k: int = 2
    tie_embeddings: bool = True  # Share input/output embeddings


# =============================================================================
# Core Components (inline for standalone use)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class RoPEAttention(nn.Module):
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

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
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

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention = RoPEAttention(
            config.d_model, config.num_heads, config.max_seq_len, config.dropout
        )
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


# =============================================================================
# Complete LLM Model
# =============================================================================

class LLM(nn.Module):
    """
    Complete Language Model with modern architecture.

    Features:
    - RoPE (Rotary Position Embeddings)
    - SwiGLU activation
    - RMSNorm
    - Pre-norm architecture
    - Optional tied embeddings

    Args:
        config: LLMConfig object

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Model Comparison (approximate params):                                   ║
    ║                                                                           ║
    ║  Our 0.1B model:     ~13M params                                          ║
    ║  GPT-2 Small:        124M params                                          ║
    ║  GPT-2 Medium:       355M params                                          ║
    ║  GPT-2 Large:        774M params                                          ║
    ║  GPT-2 XL:           1.5B params                                          ║
    ║  LLaMA 7B:           7B params                                            ║
    ║  LLaMA 13B:          13B params                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model)

        # Output projection
        if config.tie_embeddings:
            self.output_projection = None  # Will use tied embeddings
        else:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            labels: Optional target token IDs for computing loss

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Optional cross-entropy loss if labels provided
        """
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Output projection
        if self.config.tie_embeddings:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.output_projection(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with probability threshold

        Returns:
            Generated token IDs including input
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get predictions for last position
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Truncate if exceeding max length
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

        return input_ids

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            "embeddings": sum(p.numel() for p in self.token_embedding.parameters()),
            "layers": sum(p.numel() for p in self.layers.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
        }

        if self.output_projection is not None:
            counts["output"] = sum(p.numel() for p in self.output_projection.parameters())
        else:
            counts["output"] = 0  # Tied

        counts["total"] = sum(counts.values())
        if self.config.tie_embeddings:
            counts["total_unique"] = counts["total"]  # No extra output params
        else:
            counts["total_unique"] = counts["total"]

        return counts


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate the complete LLM model.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         COMPLETE LLM DEMO                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("COMPLETE LLM MODEL DEMONSTRATION")
    print("=" * 80)

    # Create model
    config = LLMConfig()
    model = LLM(config)

    print(f"\nModel Configuration:")
    print(f"  vocab_size:   {config.vocab_size:,}")
    print(f"  d_model:      {config.d_model}")
    print(f"  num_heads:    {config.num_heads}")
    print(f"  num_layers:   {config.num_layers}")
    print(f"  d_ff:         {config.d_ff}")
    print(f"  max_seq_len:  {config.max_seq_len}")

    # Parameter count
    print("\n" + "-" * 80)
    print("1. PARAMETER COUNT")
    print("-" * 80)

    param_counts = model.count_parameters()
    print(f"\nEmbeddings:  {param_counts['embeddings']:,}")
    print(f"Layers:      {param_counts['layers']:,}")
    print(f"Final Norm:  {param_counts['norm']:,}")
    print(f"Output:      {param_counts['output']:,} (tied with embeddings)")
    print(f"─" * 40)
    print(f"Total:       {param_counts['total']:,} ({param_counts['total'] / 1e6:.2f}M)")

    # Forward pass test
    print("\n" + "-" * 80)
    print("2. FORWARD PASS TEST")
    print("-" * 80)

    batch_size = 2
    seq_len = 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels)

    print(f"\nInput shape:    {input_ids.shape}")
    print(f"Logits shape:   {logits.shape}")
    print(f"Loss:           {loss.item():.4f}" if loss is not None else "Loss: None")

    # Generation test
    print("\n" + "-" * 80)
    print("3. TEXT GENERATION TEST")
    print("-" * 80)

    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Dummy prompt

    print(f"\nInput tokens: {prompt_ids[0].tolist()}")
    generated = model.generate(
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
    )
    print(f"Generated:    {generated[0].tolist()}")

    # Architecture visualization
    print("\n" + "-" * 80)
    print("4. ARCHITECTURE SUMMARY")
    print("-" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Model Architecture                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │    Input: (batch, seq_len)                                               │
    │       │                                                                  │
    │       ▼                                                                  │
    │    ┌─────────────────────────────────────┐                              │
    │    │  Token Embedding (10,000 × 256)     │  2.56M params                │
    │    └─────────────────────────────────────┘                              │
    │       │                                                                  │
    │       ▼                                                                  │
    │    ┌─────────────────────────────────────┐                              │
    │    │  Transformer Block × 12             │                              │
    │    │  ┌─────────────────────────────────┐│                              │
    │    │  │ RMSNorm → RoPE Attn (8 heads)   ││                              │
    │    │  │ RMSNorm → SwiGLU FFN (d_ff=683) ││  ~0.57M per layer            │
    │    │  └─────────────────────────────────┘│                              │
    │    └─────────────────────────────────────┘                              │
    │       │                                                                  │
    │       ▼                                                                  │
    │    ┌─────────────────────────────────────┐                              │
    │    │  Final RMSNorm                      │  256 params                  │
    │    └─────────────────────────────────────┘                              │
    │       │                                                                  │
    │       ▼                                                                  │
    │    ┌─────────────────────────────────────┐                              │
    │    │  Output (tied embeddings)           │  0 extra params              │
    │    └─────────────────────────────────────┘                              │
    │       │                                                                  │
    │       ▼                                                                  │
    │    Output: (batch, seq_len, vocab_size)                                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)

    # Comparison with larger models
    print("\n" + "-" * 80)
    print("5. MODEL SCALING COMPARISON")
    print("-" * 80)
    print("""
    ┌─────────────────────┬────────────┬────────────┬────────────┬─────────────┐
    │ Model               │ d_model    │ num_layers │ num_heads  │ Params      │
    ├─────────────────────┼────────────┼────────────┼────────────┼─────────────┤
    │ Our Model           │ 256        │ 12         │ 8          │ ~13M        │
    │ GPT-2 Small         │ 768        │ 12         │ 12         │ 124M        │
    │ GPT-2 Medium        │ 1024       │ 24         │ 16         │ 355M        │
    │ GPT-2 Large         │ 1280       │ 36         │ 20         │ 774M        │
    │ GPT-2 XL            │ 1600       │ 48         │ 25         │ 1.5B        │
    │ LLaMA 7B            │ 4096       │ 32         │ 32         │ 7B          │
    │ LLaMA 13B           │ 5120       │ 40         │ 40         │ 13B         │
    │ LLaMA 70B           │ 8192       │ 80         │ 64         │ 70B         │
    └─────────────────────┴────────────┴────────────┴────────────┴─────────────┘

    Scaling rules:
    - Params ∝ d_model² × num_layers
    - Compute ∝ d_model² × seq_len × num_layers
    """)

    # Memory requirements
    print("\n" + "-" * 80)
    print("6. MEMORY REQUIREMENTS")
    print("-" * 80)

    param_size = param_counts['total'] * 4  # float32
    param_size_mb = param_size / (1024 ** 2)

    # Gradient size (same as params)
    grad_size_mb = param_size_mb

    # Optimizer state (Adam: 2x params for momentum and variance)
    optimizer_size_mb = param_size_mb * 2

    # Activation size (rough estimate)
    activation_size_mb = batch_size * seq_len * config.d_model * config.num_layers * 4 / (1024 ** 2)

    print(f"\nParameters (FP32):     {param_size_mb:.1f} MB")
    print(f"Gradients:             {grad_size_mb:.1f} MB")
    print(f"Optimizer states:      {optimizer_size_mb:.1f} MB")
    print(f"Activations (est.):    {activation_size_mb:.1f} MB")
    print(f"─" * 40)
    print(f"Total (training):      {param_size_mb + grad_size_mb + optimizer_size_mb + activation_size_mb:.1f} MB")
    print(f"Total (inference):     {param_size_mb + activation_size_mb:.1f} MB")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Complete LLM = Embeddings + Transformer Blocks + Output Head
    2. Tied embeddings save parameters (share input/output)
    3. Modern architecture: RoPE + SwiGLU + RMSNorm + Pre-norm
    4. Generation uses autoregressive sampling
    5. Memory scales with model size and sequence length
    6. This ~13M param model is good for learning, not production

    Next: 09_training.py - Training the Model
    """)


if __name__ == "__main__":
    demo()
