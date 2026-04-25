"""
================================================================================
TRAINING: ADVANCED OPTIMIZERS & MEMORY OPTIMIZATION
================================================================================

Advanced techniques for training large models:

1. Muon Optimizer - Momentum with orthogonalization for better convergence
2. Gradient Checkpointing - Trade compute for memory

================================================================================
ILLUSTRATION: Muon Optimizer Concept
================================================================================

Standard Momentum (SGDM):
    v_t = β × v_{t-1} + (1 - β) × g_t
    θ_t = θ_{t-1} - lr × v_t

Muon (Momentum + Orthogonalization):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Gradient g_t                                                      │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────┐                                               │
    │    │ Momentum Update │                                               │
    │    │ v_t = βv + (1-β)g│                                               │
    │    └─────────────────┘                                               │
    │         │                                                            │
    │         ▼                                                            │
    │    ┌─────────────────────────────────────────────┐                   │
    │    │         Orthogonalization                    │                   │
    │    │                                              │                   │
    │    │   Newton-Schulz iteration:                   │                   │
    │    │   X' = (3X - X³) / 2  (approximates          │                   │
    │    │   orthogonalization via Taylor expansion)    │                   │
    │    │                                              │                   │
    │    │   Benefits:                                  │                   │
    │    │   - Prevents gradient collapse               │                   │
    │    │   - Better conditioning                      │                   │
    │    │   - Faster convergence                       │                   │
    │    └─────────────────────────────────────────────┘                   │
    │         │                                                            │
    │         ▼                                                            │
    │    θ_t = θ_{t-1} - lr × orthogonalized_v_t                           │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Gradient Checkpointing
================================================================================

Without checkpointing:
    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
    │ fwd │───►│ fwd │───►│ fwd │───►│ fwd │───►│ fwd │
    │  1  │    │  2  │    │  3  │    │  4  │    │  5  │
    └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
       │          │          │          │          │
     activ     activ     activ     activ     activ
       ▼          ▼          ▼          ▼          ▼
    Store all activations in memory: O(L × N × D)

With checkpointing:
    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
    │ fwd │───►│ fwd │───►│ fwd │───►│ fwd │───►│ fwd │
    │  1  │    │  2  │    │  3  │    │  4  │    │  5  │
    └──┬──┘    └─────┘    └──┬──┘    └─────┘    └─────┘
       │    (recompute)      │
     checkpoint           checkpoint
       ▼                     ▼
    Memory: O(√L × N × D) instead of O(L × N × D)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# =============================================================================
# 1. Muon Optimizer
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon Optimizer: Momentum + Orthogonalization via Newton-Schulz iteration.

    Muon orthogonalizes the momentum matrix to improve optimization dynamics.
    This helps prevent gradient collapse and improves conditioning.

    Key insight: Instead of using raw gradients, orthogonalize the update
    direction to maintain diversity in the parameter space.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.0)
        nesterov: Use Nesterov momentum (default: True)
        backend: 'newtonschulz2' or 'newtonschulz5' for orthogonalization

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Newton-Schulz Iteration:                                                 ║
    ║                                                                           ║
    ║  For approximately orthogonalizing a matrix X:                            ║
    ║                                                                           ║
    ║  X₀ = X / ||X||_F                                                         ║
    ║  X_{k+1} = (3X_k - X_k³) / 2                                              ║
    ║                                                                           ║
    ║  This converges to the orthogonal projection of X onto O(n)               ║
    ║  (the orthogonal group) for matrices with singular values < √3            ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        backend: str = "newtonschulz2",
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            backend=backend,
        )
        super().__init__(params, defaults)

    def _newtonschulz2(self, X: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration for approximate orthogonalization.

        Converges to orthogonal matrix in O(log(1/ε)) iterations.
        """
        # Normalize
        a = X.norm() + 1e-7
        X = X / a

        # Newton-Schulz iteration
        for _ in range(num_iters):
            X = (1.5 * X - 0.5 * X @ X @ X)

        return X

    def _newtonschulz5(self, X: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """
        Higher-order Newton-Schulz iteration (faster convergence).
        """
        # Normalize
        a = X.norm() + 1e-7
        X = X / a

        # 5th order iteration
        for _ in range(num_iters):
            X2 = X @ X
            X4 = X2 @ X2
            X = X @ (1.875 * torch.eye(X.shape[0], device=X.device)
                     - 1.25 * X2 + 0.375 * X4)

        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            backend = group["backend"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                # Get or initialize momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]

                # Apply momentum
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Nesterov momentum
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Orthogonalize update for 2D parameters (weight matrices)
                if update.dim() == 2 and min(update.shape) >= 2:
                    if backend == "newtonschulz2":
                        update = self._newtonschulz2(update)
                    elif backend == "newtonschulz5":
                        update = self._newtonschulz5(update)

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss


# =============================================================================
# 2. Gradient Checkpointing
# =============================================================================

# Simplified RoPE Attention for checkpointing demo
class _RoPEAttention(nn.Module):
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

        self.attention = _RoPEAttention(d_model, num_heads, max_seq_len)
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


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate advanced optimizers and checkpointing."""
    print("=" * 80)
    print("ADVANCED OPTIMIZERS & MEMORY OPTIMIZATION")
    print("=" * 80)

    # Muon Optimizer Demo
    print("\n" + "-" * 80)
    print("1. MUON OPTIMIZER")
    print("-" * 80)

    # Create a simple model
    model = nn.Linear(64, 64)

    # Compare standard AdamW vs Muon
    print("\nOptimizer comparison on a simple linear layer:")

    # Standard AdamW
    model_adamw = nn.Linear(64, 64)
    optimizer_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=0.01)

    # Muon
    model_muon = nn.Linear(64, 64)
    model_muon.load_state_dict(model_adamw.state_dict())  # Same init
    optimizer_muon = Muon(model_muon.parameters(), lr=0.01)

    # Simple training loop
    x = torch.randn(32, 64)
    y = torch.randn(32, 64)

    print("\nTraining for 50 steps...")
    losses_adamw = []
    losses_muon = []

    for step in range(50):
        # AdamW
        optimizer_adamw.zero_grad()
        out = model_adamw(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer_adamw.step()
        losses_adamw.append(loss.item())

        # Muon
        optimizer_muon.zero_grad()
        out = model_muon(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer_muon.step()
        losses_muon.append(loss.item())

    print(f"\nAdamW final loss: {losses_adamw[-1]:.4f}")
    print(f"Muon final loss:  {losses_muon[-1]:.4f}")

    print("""
    Muon Benefits:
    - Orthogonalized updates prevent gradient collapse
    - Better conditioning for optimization
    - Works particularly well for 2D weight matrices
    - Can lead to faster convergence on some tasks
    """)

    # Gradient Checkpointing Demo
    print("-" * 80)
    print("2. GRADIENT CHECKPOINTING")
    print("-" * 80)

    checkpointed_block = CheckpointedTransformerBlock(
        d_model=64, d_ff=128, num_heads=4, max_seq_len=128, use_checkpoint=True
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
    print("TRAINING OPTIMIZATION SUMMARY")
    print("-" * 80)
    print("""
    ┌────────────────────────┬─────────────────────┬──────────────────────┐
    │ Technique              │ Benefit             │ Trade-off            │
    ├────────────────────────┼─────────────────────┼──────────────────────┤
    │ Muon Optimizer         │ Faster convergence  │ Extra computation    │
    │ Gradient Checkpointing │ O(L) → O(√L) memory │ +33% recomputation   │
    │ Mixed Precision        │ 2x faster, 1/2 mem  │ Slight precision loss│
    └────────────────────────┴─────────────────────┴──────────────────────┘
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Muon orthogonalizes momentum for better optimization dynamics
    2. Gradient checkpointing trades compute for memory in large models
    3. Both techniques enable training larger models with limited resources

    Next: inference/inference_optimization.py - Flash Attention & KV Cache
    """)


if __name__ == "__main__":
    demo()
