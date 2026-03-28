"""
================================================================================
LLM Learning Module 6: MIXTURE OF EXPERTS (MoE)
================================================================================

What is Mixture of Experts?
---------------------------
MoE replaces the dense feed-forward network with multiple "expert" networks.
A gating mechanism routes each token to a subset of experts, allowing
the model to scale to huge sizes while keeping computation manageable.

Key Concepts:
1. Experts: Multiple parallel feed-forward networks
2. Router/Gating: Decides which experts to use for each token
3. Top-k Routing: Only use top k experts (typically k=1 or k=2)
4. Load Balancing: Ensure all experts are used evenly

================================================================================
ILLUSTRATION: Dense vs Sparse MoE
================================================================================

Dense FFN (Standard Transformer):
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │    Token ──► ┌───────────────────┐ ──► Output                   │
    │              │   Single FFN      │                               │
    │              │   (all params)    │                               │
    │              └───────────────────┘                               │
    │                                                                  │
    │    All tokens use ALL parameters                                 │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

Sparse MoE:
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │           ┌──► Expert 1 ──┐                                      │
    │           │               │                                      │
    │    Token ─┼──► Expert 2 ──┼── (weighted sum) ──► Output         │
    │     │     │               │                                      │
    │     │     └──► Expert 3 ──┘                                      │
    │     │         ...                                                │
    │     └──► Expert N                                                │
    │                                                                  │
    │    Router selects Top-k experts per token                        │
    │    Only k/N of parameters used per forward pass!                 │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Top-k Routing Example
================================================================================

    Tokens:    [T1,  T2,  T3,  T4]
    Experts:   [E1,  E2,  E3,  E4,  E5,  E6,  E7,  E8]

    Router scores (T1): [0.1, 0.3, 0.9, 0.2, 0.8, 0.05, 0.15, 0.4]

    Top-2 selection: E3 (0.9) and E5 (0.8)
    ────────────────────────────────────────────
    T1 ──► E3 (weight 0.53) ──► 0.53 × E3(T1) ──┐
          E5 (weight 0.47) ──► 0.47 × E5(T1) ──┼──► Sum = Output for T1
                                               ┘

    Benefits:
    - 8 experts = 8× parameters, but only 2× computation
    - Model can specialize experts for different tasks/domains

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Expert(nn.Module):
    """
    Single Expert Network (Feed-Forward)

    Each expert is a standard FFN (can use SwiGLU, GELU, etc.)

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq, d_model) or (num_tokens, d_model)

        Returns:
            Output of same shape
        """
        return self.dropout(self.w2(F.silu(self.w1(x))))


class Router(nn.Module):
    """
    Token Router for MoE

    Computes routing weights for each token-expert pair.

    Args:
        d_model: Input dimension
        num_experts: Number of experts
        noise_std: Standard deviation of noise for load balancing (training only)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Load Balancing with Noise:                                               ║
    ║                                                                           ║
    ║  During training, we add random noise to routing scores:                  ║
    ║    scores = W_router(x) + noise                                           ║
    ║                                                                           ║
    ║  This encourages exploration and prevents always picking same experts     ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_experts: int, noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and indices.

        Args:
            x: Input of shape (batch, seq, d_model)
            training: Whether in training mode (adds noise)

        Returns:
            weights: Softmax weights for selected experts (batch, seq, top_k)
            indices: Expert indices (batch, seq, top_k)
        """
        # Compute raw scores
        scores = self.linear(x)  # (batch, seq, num_experts)

        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(scores) * self.noise_std
            scores = scores + noise

        return scores


class MoE(nn.Module):
    """
    Mixture of Experts Layer

    Implements sparse MoE with top-k routing.

    Args:
        d_model: Model dimension
        d_ff: Expert hidden dimension
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        dropout: Dropout probability
        noise_std: Noise for load balancing

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Scaling Properties:                                                      ║
    ║                                                                           ║
    ║  Parameters: O(num_experts × d_model × d_ff)                              ║
    ║  Compute:   O(top_k × d_model × d_ff)  (independent of num_experts!)     ║
    ║                                                                           ║
    ║  Example: Switch Transformer (Google)                                     ║
    ║  - 2048 experts, top_k = 1                                                ║
    ║  - 1.6T parameters, but only 2B active per forward pass                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Router
        self.router = Router(d_model, num_experts, noise_std)

        # For load balancing loss
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with top-k routing.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
        num_tokens = x_flat.shape[0]

        # Get routing scores
        scores = self.router(x_flat, self.training)  # (num_tokens, num_experts)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        # (num_tokens, top_k)

        # Normalize scores with softmax
        top_k_weights = F.softmax(top_k_scores, dim=-1)  # (num_tokens, top_k)

        # Compute output by routing to selected experts
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # top_k_indices: (num_tokens, top_k)
            expert_mask = (top_k_indices == expert_idx)  # (num_tokens, top_k)

            # Get tokens for this expert
            token_indices, k_indices = torch.where(expert_mask)

            if token_indices.numel() == 0:
                continue

            # Get tokens
            expert_input = x_flat[token_indices]  # (num_tokens_for_expert, d_model)

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for these tokens
            weights = top_k_weights[token_indices, k_indices]  # (num_tokens_for_expert,)

            # Weight and accumulate
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        # Compute auxiliary loss for load balancing
        if self.training:
            self._compute_aux_loss(scores, top_k_indices)

        # Reshape back to (batch, seq, d_model)
        output = output.view(batch_size, seq_len, d_model)

        return output

    def _compute_aux_loss(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Compute load balancing auxiliary loss.

        This loss encourages even distribution of tokens across experts.

        Formula: aux_loss = num_experts × sum(f_i × P_i)
        where:
            f_i = fraction of tokens routed to expert i
            P_i = average routing probability for expert i
        """
        # Compute fraction of tokens per expert
        # One-hot encode selected experts
        num_tokens = scores.shape[0]
        one_hot = F.one_hot(indices.view(-1), self.num_experts).float()
        # (num_tokens * top_k, num_experts)

        tokens_per_expert = one_hot.sum(dim=0) / (num_tokens * self.top_k)
        # (num_experts,)

        # Compute average routing probability per expert
        route_prob = F.softmax(scores, dim=-1).mean(dim=0)  # (num_experts,)

        # Auxiliary loss
        self.aux_loss = self.num_experts * torch.sum(tokens_per_expert * route_prob)


class SwitchMoE(nn.Module):
    """
    Switch Transformer Style MoE (Top-1 Routing)

    Simplified version that routes each token to exactly one expert.

    Args:
        d_model: Model dimension
        d_ff: Expert hidden dimension
        num_experts: Number of experts
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Switch Transformer (Google, 2021):                                       ║
    ║                                                                           ║
    ║  Key innovations:                                                         ║
    ║  1. Top-1 routing (simpler, faster)                                       ║
    ║  2. Expert capacity factor (limit tokens per expert)                      ║
    ║  3. Simplified load balancing loss                                        ║
    ║                                                                           ║
    ║  Results: 4x faster pre-training than T5 at same compute budget           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        x_flat = x.view(-1, d_model)

        # Get routing probabilities
        route_probs = F.softmax(self.router(x_flat), dim=-1)  # (num_tokens, num_experts)

        # Select expert with highest probability
        expert_indices = route_probs.argmax(dim=-1)  # (num_tokens,)

        # Compute output
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            token_mask = (expert_indices == expert_idx)
            if token_mask.sum() == 0:
                continue

            expert_input = x_flat[token_mask]
            expert_output = self.experts[expert_idx](expert_input)
            output[token_mask] = expert_output

        # Load balancing loss
        if self.training:
            # Fraction of tokens per expert
            f = F.one_hot(expert_indices, self.num_experts).float().mean(dim=0)
            # Average routing probability per expert
            P = route_probs.mean(dim=0)
            # Auxiliary loss
            self.aux_loss = self.num_experts * torch.sum(f * P)

        return output.view(batch_size, seq_len, d_model)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate Mixture of Experts.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                          MoE DEMO                                         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("MIXTURE OF EXPERTS (MoE) DEMONSTRATION")
    print("=" * 80)

    # Hyperparameters
    d_model = 64
    d_ff = 128
    num_experts = 8
    top_k = 2
    batch_size = 2
    seq_len = 8

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Config: d_model={d_model}, d_ff={d_ff}, num_experts={num_experts}, top_k={top_k}")

    # Standard FFN for comparison
    print("\n" + "-" * 80)
    print("1. STANDARD DENSE FFN")
    print("-" * 80)

    dense_ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.SiLU(),
        nn.Linear(d_ff, d_model),
    )
    dense_params = sum(p.numel() for p in dense_ffn.parameters())
    dense_out = dense_ffn(x)

    print(f"Parameters: {dense_params:,}")
    print(f"Output shape: {dense_out.shape}")
    print(f"Active params per forward: {dense_params:,} (100%)")

    # MoE with Top-k routing
    print("\n" + "-" * 80)
    print("2. MoE WITH TOP-k ROUTING")
    print("-" * 80)

    moe = MoE(d_model, d_ff, num_experts, top_k)
    moe_params = sum(p.numel() for p in moe.parameters())
    moe_out = moe(x)

    active_params = (d_model * d_ff + d_ff * d_model) * top_k + d_model * num_experts

    print(f"Total parameters: {moe_params:,}")
    print(f"Output shape: {moe_out.shape}")
    print(f"Active params per forward: ~{active_params:,} ({top_k}/{num_experts} experts)")
    print(f"Auxiliary loss: {moe.aux_loss.item():.4f}")

    # Routing visualization
    print("\n" + "-" * 80)
    print("3. ROUTING VISUALIZATION")
    print("-" * 80)

    # Get routing info
    x_flat = x.view(-1, d_model)
    scores = moe.router(x_flat, training=False)
    top_k_scores, top_k_indices = torch.topk(scores, top_k, dim=-1)

    print(f"\nRouting for first 6 tokens:")
    print("Token │  Expert IDs  │  Weights (after softmax)")
    print("──────┼──────────────┼─────────────────────────────")
    for i in range(min(6, seq_len * batch_size)):
        experts = top_k_indices[i].tolist()
        weights = F.softmax(top_k_scores[i], dim=-1).tolist()
        print(f"  {i}   │  {experts}  │  [{weights[0]:.3f}, {weights[1]:.3f}]")

    # Expert utilization
    print("\n" + "-" * 80)
    print("4. EXPERT UTILIZATION")
    print("-" * 80)

    expert_counts = torch.zeros(num_experts)
    for idx in top_k_indices.flatten():
        expert_counts[idx] += 1

    print("\nExpert usage distribution:")
    for i, count in enumerate(expert_counts):
        bar = "█" * int(count * 2)
        print(f"  Expert {i}: {bar} ({int(count)} tokens)")

    # Scaling comparison
    print("\n" + "-" * 80)
    print("5. SCALING COMPARISON (LLM Scale)")
    print("-" * 80)

    # Simulate larger scale
    d_model_large = 2048
    d_ff_large = 8192

    print(f"\nConfig: d_model={d_model_large}, d_ff={d_ff_large}")
    print("─" * 60)

    # Dense
    dense_params_large = d_model_large * d_ff_large + d_ff_large * d_model_large
    print(f"Dense FFN:       {dense_params_large / 1e6:,.0f}M params, 100% compute")

    # MoE with different expert counts
    for num_exp in [8, 64, 512]:
        moe_params_large = num_exp * (d_model_large * d_ff_large + d_ff_large * d_model_large)
        moe_compute = 2 * (d_model_large * d_ff_large + d_ff_large * d_model_large)  # top-2
        efficiency = moe_compute / moe_params_large * 100
        print(f"MoE ({num_exp:3d} exp):  {moe_params_large / 1e9:,.1f}B params, {efficiency:.1f}% compute")

    print("""
    Key insight: MoE can scale to massive parameter counts while keeping
    computation constant. A 1.6T parameter model can have similar inference
    speed to a 7B dense model!
    """)

    # Switch MoE
    print("-" * 80)
    print("6. SWITCH MoE (TOP-1 ROUTING)")
    print("-" * 80)

    switch_moe = SwitchMoE(d_model, d_ff, num_experts)
    switch_out = switch_moe(x)

    print(f"\nTotal parameters: {sum(p.numel() for p in switch_moe.parameters()):,}")
    print(f"Output shape: {switch_out.shape}")
    print(f"Auxiliary loss: {switch_moe.aux_loss.item():.4f}")

    # Load balancing explanation
    print("\n" + "-" * 80)
    print("7. LOAD BALANCING LOSS")
    print("-" * 80)
    print("""
    The auxiliary loss ensures all experts are used evenly:

    Loss = num_experts × Σ(f_i × P_i)

    where:
        f_i = fraction of tokens routed to expert i
        P_i = average routing probability for expert i

    Example with 8 experts, 100 tokens:
    ────────────────────────────────────────────────────────────────────────
    Expert │ Tokens  │ f_i    │ P_i   │ Contribution
    ───────┼─────────┼────────┼───────┼──────────────
       0   │   25    │ 0.25   │ 0.15  │ 0.0375
       1   │   15    │ 0.15   │ 0.12  │ 0.0180
       2   │   10    │ 0.10   │ 0.10  │ 0.0100  ← balanced
       3   │    5    │ 0.05   │ 0.08  │ 0.0040
       ...

    Minimizing this loss pushes f_i ≈ P_i ≈ 1/num_experts for all experts.
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. MoE replaces dense FFN with multiple expert networks
    2. Router selects top-k experts per token (typically k=1 or k=2)
    3. Scales parameters without increasing computation
    4. Load balancing loss ensures all experts are utilized
    5. Used in Switch Transformer, Mixtral, GPT-4 (rumored), etc.
    6. Trade-off: More VRAM needed, but faster inference

    Real-world examples:
    - Mixtral 8x7B: 47B params, 13B active per token (top-2)
    - Switch Transformer: 1.6T params, 2B active per token

    Next: 07_megablock.py - Efficient MoE Implementation
    """)


if __name__ == "__main__":
    demo()
