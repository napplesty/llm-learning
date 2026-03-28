"""
================================================================================
LLM Learning Module 5: SwiGLU ACTIVATION
================================================================================

What is SwiGLU?
---------------
SwiGLU is a variant of the Gated Linear Unit (GLU) that uses Swish (SiLU)
as the activation function. It's used in LLaMA, PaLM, and other modern LLMs.

Key Concepts:
1. GLU: Gated Linear Unit - uses a gate to control information flow
2. Swish/SiLU: Sigmoid Linear Unit - x * sigmoid(x)
3. SwiGLU: Combines both for better performance

================================================================================
ILLUSTRATION: Feed-Forward Network Variants
================================================================================

Standard FFN (GPT-2, BERT):
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │    Input ──► Linear ──► GELU/ReLU ──► Linear ──► Output         │
    │             (d→4d)                (4d→d)                         │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Parameters: d×4d + 4d×d = 8d²

GLU-based FFN:
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │         ┌──► Linear (d→4d) ──► Sigmoid ──┐                       │
    │    Input│                               │ × ──► Linear ──► Output│
    │         └──► Linear (d→4d) ─────────────┘    (4d→d)              │
    │               (gate)        (value)                              │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

    Parameters: 2×(d×4d) + 4d×d = 12d²

SwiGLU FFN (LLaMA):
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │         ┌──► Linear (d→d_ff) ──► Swish ──┐                       │
    │    Input│                               │ × ──► Linear ──► Output│
    │         └──► Linear (d→d_ff) ───────────┘    (d_ff→d)            │
    │                                                                  │
    │    Where: Swish(x) = x × sigmoid(x)                             │
    │           d_ff = (2/3) × 4d ≈ 2.67d (adjusted to keep params same)│
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Why Gating Works
================================================================================

The gate controls how much information passes through:

    Input value:  [0.5, 1.0, -0.3, 2.0]
    Gate output:  [0.9, 0.1, 0.5, 0.0]  (after sigmoid, range 0-1)
    Result:       [0.45, 0.1, -0.15, 0.0]  (element-wise multiplication)

    - Some dimensions are "opened" (gate ≈ 1)
    - Some dimensions are "closed" (gate ≈ 0)
    - The network learns which dimensions to use for each input

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    """
    Swish / SiLU (Sigmoid Linear Unit) Activation Function

    Formula: Swish(x) = x × sigmoid(x)

    Properties:
    - Smooth and non-monotonic (unlike ReLU)
    - Self-gated (the sigmoid acts as a gate)
    - Negative values are preserved (unlike ReLU)
    - Consistently outperforms ReLU in deep networks

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Activation Comparison:                                                    ║
    ║                                                                           ║
    ║  ReLU:     max(0, x)              - Simple, but kills negative gradients  ║
    ║  GELU:     x × Φ(x)               - Smooth, used in BERT/GPT-2            ║
    ║  Swish:    x × σ(x)               - Smooth, outperforms ReLU in deep nets ║
    ║  SwiGLU:   Swish(gate) × value    - Best of both worlds                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation

    Used in some transformer variants.

    Formula: GEGLU(x) = GELU(W₁x) × (W₂x)

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (feed-forward)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.w_gate(x))
        value = self.w_value(x)
        return self.dropout(self.w_out(gate * value))


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network

    Used in LLaMA, PaLM, and other modern LLMs.

    Formula: SwiGLU(x) = (Swish(W_gate x) ⊙ W_value x) W_out

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 8/3 × d_model for same param count)
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Parameter Count Comparison (for d_model = 768):                          ║
    ║                                                                           ║
    ║  Standard FFN (d_ff = 4×d):                                               ║
    ║    768 × 3072 + 3072 × 768 = 4,718,592 params                             ║
    ║                                                                           ║
    ║  SwiGLU FFN (d_ff = 8/3 × d ≈ 2048):                                      ║
    ║    768 × 2048 × 2 + 2048 × 768 = 4,718,592 params (same!)                 ║
    ║                                                                           ║
    ║  LLaMA uses d_ff = 2.67 × d_model to match parameter counts               ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Compute gate and value projections
        gate = self.w_gate(x)  # (batch, seq, d_ff)
        value = self.w_up(x)   # (batch, seq, d_ff)

        # Apply Swish to gate and multiply with value
        # Swish(x) = x * sigmoid(x)
        hidden = F.silu(gate) * value  # (batch, seq, d_ff)

        # Project back to d_model
        output = self.w_down(hidden)  # (batch, seq, d_model)

        return self.dropout(output)


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network with GELU

    Used in GPT-2, BERT, and original transformer.

    Formula: FFN(x) = GELU(x W₁ + b₁) W₂ + b₂

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 × d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate SwiGLU and compare with other FFN variants.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        SwiGLU DEMO                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("SwiGLU ACTIVATION DEMONSTRATION")
    print("=" * 80)

    # Hyperparameters
    d_model = 64
    batch_size = 2
    seq_len = 8

    # Calculate d_ff for SwiGLU to match standard FFN parameter count
    # Standard: d × 4d + 4d × d = 8d²
    # SwiGLU: 2 × d × d_ff + d_ff × d = 3 × d × d_ff
    # To match: d_ff = 8d² / 3d = 8d/3 ≈ 2.67d
    d_ff_standard = 4 * d_model  # 256
    d_ff_swiglu = int(8 * d_model / 3)  # 170 (rounded)

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"d_model: {d_model}")
    print(f"d_ff (standard): {d_ff_standard}")
    print(f"d_ff (SwiGLU): {d_ff_swiglu}")

    # Activation Function Visualization
    print("\n" + "-" * 80)
    print("1. ACTIVATION FUNCTION COMPARISON")
    print("-" * 80)

    test_vals = torch.linspace(-3, 3, 7)

    print("\nInput values: " + "  ".join([f"{v:6.1f}" for v in test_vals.tolist()]))
    print("─" * 60)

    relu_out = F.relu(test_vals)
    print(f"ReLU:        " + "  ".join([f"{v:6.2f}" for v in relu_out.tolist()]))

    gelu_out = F.gelu(test_vals)
    print(f"GELU:        " + "  ".join([f"{v:6.2f}" for v in gelu_out.tolist()]))

    swish_out = F.silu(test_vals)  # SiLU = Swish
    print(f"Swish/SiLU:  " + "  ".join([f"{v:6.2f}" for v in swish_out.tolist()]))

    sigmoid_out = torch.sigmoid(test_vals)
    print(f"Sigmoid:     " + "  ".join([f"{v:6.2f}" for v in sigmoid_out.tolist()]))

    print("""
    Key differences:
    - ReLU: Hard cutoff at 0, no negative values
    - GELU: Smooth version of ReLU, preserves small negative values
    - Swish: Non-monotonic, has a dip for negative values
    - Sigmoid: Always between 0 and 1, used as gate
    """)

    # Standard FFN
    print("-" * 80)
    print("2. STANDARD FFN (GPT-2 Style)")
    print("-" * 80)

    std_ffn = StandardFFN(d_model, d_ff_standard)
    std_out = std_ffn(x)

    print(f"\nOutput shape: {std_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in std_ffn.parameters()):,}")

    # SwiGLU FFN
    print("\n" + "-" * 80)
    print("3. SwiGLU FFN (LLaMA Style)")
    print("-" * 80)

    swiglu_ffn = SwiGLU(d_model, d_ff_swiglu)
    swiglu_out = swiglu_ffn(x)

    print(f"\nOutput shape: {swiglu_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in swiglu_ffn.parameters()):,}")

    # Parameter comparison
    print("\n" + "-" * 80)
    print("4. PARAMETER COMPARISON (d_model = 768, like GPT-2 small)")
    print("-" * 80)

    d_model_real = 768
    d_ff_std = 4 * d_model_real  # 3072
    d_ff_swiglu = int(8 * d_model_real / 3)  # 2048

    std_params = d_model_real * d_ff_std + d_ff_std * d_model_real
    swiglu_params = 2 * d_model_real * d_ff_swiglu + d_ff_swiglu * d_model_real

    print(f"\nStandard FFN (d_ff = {d_ff_std}):")
    print(f"  Params = {d_model_real} × {d_ff_std} + {d_ff_std} × {d_model_real}")
    print(f"         = {std_params:,}")

    print(f"\nSwiGLU FFN (d_ff = {d_ff_swiglu}):")
    print(f"  Params = 2 × {d_model_real} × {d_ff_swiglu} + {d_ff_swiglu} × {d_model_real}")
    print(f"         = {swiglu_params:,}")

    print(f"\nDifference: {abs(std_params - swiglu_params):,} params ({abs(std_params - swiglu_params) / std_params * 100:.1f}%)")

    # Gating visualization
    print("\n" + "-" * 80)
    print("5. GATING MECHANISM VISUALIZATION")
    print("-" * 80)

    # Create a simple visualization
    print("""
    How the gate works in SwiGLU:

    Input:     [0.5,  1.0, -0.3,  2.0, -1.5,  0.0]
                    ↓
    W_gate:    [0.8,  1.2, -0.5,  2.5, -2.0,  0.3]
                    ↓
    Swish:     [0.6,  0.9, -0.1,  2.3, -0.4,  0.0]  ← gate values
                    ×
    W_up:      [1.0,  0.5,  0.8,  1.5, -0.5,  0.2]  ← value values
                    ‖
    Result:    [0.6,  0.45, -0.08, 3.45, 0.2,  0.0]

    The gate learns to:
    - Open (→1) for important features
    - Close (→0) for irrelevant features
    - Handle negative values smoothly
    """)

    # Performance comparison (theoretical)
    print("\n" + "-" * 80)
    print("6. WHY SwiGLU PERFORMS BETTER")
    print("-" * 80)
    print("""
    Research findings (from "GLU Variants Improve Transformer"):

    ┌─────────────────┬────────────────┬─────────────────┐
    │ Activation      │ Params         │ Performance     │
    ├─────────────────┼────────────────┼─────────────────┤
    │ ReLU            │ Baseline       │ Baseline        │
    │ GELU            │ Baseline       │ +0.1-0.2%       │
    │ Swish           │ Baseline       │ +0.1-0.3%       │
    │ ReGLU           │ +50%           │ +0.3-0.5%       │
    │ GeGLU           │ +50%           │ +0.4-0.6%       │
    │ SwiGLU          │ +50%           │ +0.5-0.7% ★     │
    └─────────────────┴────────────────┴─────────────────┘

    Key benefits of SwiGLU:
    1. Gating allows selective feature activation
    2. Swish provides smooth gradients (no dead neurons)
    3. Non-monotonic behavior helps optimization
    4. Better scaling to larger models
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. SwiGLU = Swish(Gate) × Value, combines gating with smooth activation
    2. Gating allows the network to selectively activate features
    3. Swish (x × sigmoid(x)) is smoother than ReLU with no dead neurons
    4. Use d_ff = 8/3 × d_model to match standard FFN parameter count
    5. SwiGLU consistently outperforms standard FFN in LLMs
    6. Used in LLaMA, PaLM, and many modern language models

    Next: 06_moe.py - Mixture of Experts
    """)


if __name__ == "__main__":
    demo()
