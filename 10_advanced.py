"""
================================================================================
LLM Learning Module 10: ADVANCED TOPICS
================================================================================

This module covers advanced techniques used in modern LLMs:
1. Muon Optimizer - Momentum with orthogonalization
2. CLIP-style Contrastive Learning
3. Additional modern techniques overview

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
ILLUSTRATION: CLIP (Contrastive Language-Image Pre-training)
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          CLIP Architecture                               │
    │                                                                          │
    │    Image Encoder                    Text Encoder                         │
    │    ┌───────────────┐               ┌───────────────┐                    │
    │    │               │               │               │                    │
    │    │   Image 1     │               │  "A dog"      │                    │
    │    │     │         │               │     │         │                    │
    │    │     ▼         │               │     ▼         │                    │
    │    │  Vision       │               │   Text        │                    │
    │    │  Transformer  │               │  Transformer  │                    │
    │    │     │         │               │     │         │                    │
    │    │     ▼         │               │     ▼         │                    │
    │    │  I_1 (512d)   │               │  T_1 (512d)   │                    │
    │    │               │               │               │                    │
    │    └───────────────┘               └───────────────┘                    │
    │                                                                          │
    │    ┌───────────────────────────────────────────────────────────────────┐│
    │    │                    Contrastive Loss                                ││
    │    │                                                                    ││
    │    │    Similarity Matrix (I × T^T):                                    ││
    │    │                                                                    ││
    │    │              T_1    T_2    T_3    T_4                              ││
    │    │         ┌────────────────────────────┐                             ││
    │    │    I_1  │ ████  0.2   0.1   0.3     │  ← Maximize diagonal        ││
    │    │    I_2  │  0.1  ████  0.2   0.1     │     Minimize off-diagonal   ││
    │    │    I_3  │  0.2   0.1  ████  0.2     │                             ││
    │    │    I_4  │  0.3   0.2   0.1  ████    │                             ││
    │    │         └────────────────────────────┘                             ││
    │    │                                                                    ││
    │    │    Loss = CrossEntropy(sim_matrix, identity)                      ││
    │    └───────────────────────────────────────────────────────────────────┘│
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# =============================================================================
# Muon Optimizer
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
# CLIP-style Contrastive Learning
# =============================================================================

class CLIPStyleEncoder(nn.Module):
    """
    CLIP-style encoder for contrastive learning.

    Projects embeddings to a shared space for computing similarities.

    Args:
        embed_dim: Input embedding dimension
        projection_dim: Output projection dimension (shared space)
    """

    def __init__(self, embed_dim: int, projection_dim: int = 512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings and normalize.

        Args:
            x: Input embeddings (batch, embed_dim)

        Returns:
            Normalized projections (batch, projection_dim)
        """
        projected = self.projection(x)
        return F.normalize(projected, dim=-1)


def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CLIP-style contrastive loss.

    The loss encourages matching image-text pairs to have high similarity
    while non-matching pairs have low similarity.

    Args:
        image_features: Normalized image features (batch, dim)
        text_features: Normalized text features (batch, dim)
        logit_scale: Learnable scale parameter
        temperature: Temperature for scaling logits

    Returns:
        loss: Combined contrastive loss
        accuracy: Top-1 accuracy

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Contrastive Loss Formulation:                                            ║
    ║                                                                           ║
    ║  L = (L_i2t + L_t2i) / 2                                                  ║
    ║                                                                           ║
    ║  where:                                                                   ║
    ║    L_i2t = -1/N Σ_i log[exp(sim(i_i, t_i) / τ) / Σ_j exp(sim(i_i, t_j) / τ)]║
    ║    L_t2i = -1/N Σ_i log[exp(sim(t_i, i_i) / τ) / Σ_j exp(sim(t_i, i_j) / τ)]║
    ║                                                                           ║
    ║  This creates a symmetric loss that learns from both directions.          ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    batch_size = image_features.shape[0]

    # Compute similarity matrix
    # (batch, batch) - each entry is sim(image_i, text_j)
    logits = (image_features @ text_features.T) * torch.exp(logit_scale) / temperature

    # Create labels (diagonal = correct pairs)
    labels = torch.arange(batch_size, device=logits.device)

    # Cross-entropy loss (both directions)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2

    # Compute accuracy
    with torch.no_grad():
        pred_i2t = logits.argmax(dim=1)
        pred_t2i = logits.T.argmax(dim=1)
        accuracy = (pred_i2t == labels).float().mean()

    return loss, accuracy


# =============================================================================
# Additional Modern Techniques Overview
# =============================================================================

def print_techniques_overview():
    """Print overview of modern LLM techniques."""

    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                     MODERN LLM TECHNIQUES OVERVIEW                              ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  1. ARCHITECTURE IMPROVEMENTS                                                   ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • RoPE (Rotary Position Embeddings)    - Better length extrapolation           ║
║  • SwiGLU                               - Smoother activation, better grads     ║
║  • RMSNorm                              - Simpler, faster than LayerNorm        ║
║  • Grouped Query Attention (GQA)        - Efficient KV cache                    ║
║  • Sliding Window Attention             - Handle long contexts                  ║
║  • Flash Attention                      - Memory-efficient attention            ║
║                                                                                 ║
║  2. SCALING TECHNIQUES                                                          ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • Mixture of Experts (MoE)             - Scale params, not compute             ║
║  • Mixture of Depths                    - Dynamic computation per layer         ║
║  • Mixture-of-Depths-and-Experts        - Combine both approaches               ║
║                                                                                 ║
║  3. TRAINING IMPROVEMENTS                                                       ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • Muon Optimizer                       - Orthogonalized momentum               ║
║  • LION Optimizer                       - Simplified AdamW alternative          ║
║  • Sophia Optimizer                     - Second-order for LLMs                 ║
║  • Pre-normalization                    - Stable deep network training          ║
║  • QK-LayerNorm                         - Stabilize attention                   ║
║                                                                                 ║
║  4. EFFICIENCY TECHNIQUES                                                       ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • Quantization (INT8/INT4)             - Reduce memory, faster inference       ║
║  • KV Cache Compression                 - Handle longer contexts                ║
║  • Speculative Decoding                 - Faster generation                     ║
║  • Continuous Batching                  - Better GPU utilization                ║
║                                                                                 ║
║  5. MULTIMODAL TECHNIQUES                                                       ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • CLIP Contrastive Learning            - Align vision & language               ║
║  • Vision-Language Models               - GPT-4V, LLaVA, etc.                   ║
║  • Cross-Modal Attention                - Fuse different modalities             ║
║                                                                                 ║
║  6. ALIGNMENT TECHNIQUES                                                        ║
║  ───────────────────────────────────────────────────────────────────────────    ║
║  • RLHF (Reinforcement Learning)        - Human feedback alignment              ║
║  • DPO (Direct Preference Optimization) - Simpler than RLHF                     ║
║  • Constitutional AI                    - Self-critique and improve             ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate advanced techniques.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                      ADVANCED TOPICS DEMO                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("ADVANCED TOPICS DEMONSTRATION")
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

    # CLIP Demo
    print("-" * 80)
    print("2. CLIP-STYLE CONTRASTIVE LEARNING")
    print("-" * 80)

    batch_size = 8
    embed_dim = 256
    projection_dim = 128

    # Create encoders
    image_encoder = CLIPStyleEncoder(embed_dim, projection_dim)
    text_encoder = CLIPStyleEncoder(embed_dim, projection_dim)

    # Simulate embeddings
    image_embeddings = torch.randn(batch_size, embed_dim)
    text_embeddings = torch.randn(batch_size, embed_dim)

    # Encode
    image_features = image_encoder(image_embeddings)
    text_features = text_encoder(text_embeddings)

    print(f"\nImage features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")

    # Compute loss
    logit_scale = torch.tensor(0.07)
    loss, accuracy = contrastive_loss(image_features, text_features, logit_scale)

    print(f"\nContrastive loss: {loss.item():.4f}")
    print(f"Top-1 accuracy: {accuracy.item() * 100:.1f}%")

    # Show similarity matrix
    with torch.no_grad():
        sim_matrix = (image_features @ text_features.T)
        print("\nSimilarity matrix (first 4x4):")
        print("─" * 40)
        for i in range(min(4, batch_size)):
            row = sim_matrix[i, :4].tolist()
            print("  " + "  ".join([f"{s:6.3f}" for s in row]))

    print("""
    CLIP Training Tips:
    - Use large batch sizes (thousands of pairs)
    - Temperature scaling is crucial
    - Data augmentation for images helps
    - Learnable logit_scale improves performance
    """)

    # Techniques Overview
    print("\n" + "-" * 80)
    print("3. MODERN LLM TECHNIQUES OVERVIEW")
    print("-" * 80)
    print_techniques_overview()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Muon orthogonalizes momentum for better optimization
    2. CLIP aligns vision and language via contrastive learning
    3. Modern LLMs combine many techniques for best performance
    4. Efficiency techniques (quantization, caching) enable deployment
    5. Alignment techniques ensure safe and helpful outputs

    This completes the LLM learning modules! You now have:
    - Tokenization (BPE)
    - Embeddings (Token + Position)
    - Attention (Multi-head, RoPE, GQA)
    - FFN (SwiGLU, MoE)
    - Complete Model Architecture
    - Training Pipeline
    - Advanced Techniques

    Next: run_all.py - Interactive demo of all components
    """)


if __name__ == "__main__":
    demo()
