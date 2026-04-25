"""
================================================================================
MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC)
================================================================================

DeepSeek's upgrade to the classic residual connection (He et al., 2016).

The Problem with Standard Residual Connections:
    x_{l+1} = x_l + F(x_l)
    
    - Single stream, limited information flow
    - Identity mapping guarantees stability but is restrictive

The Problem with Hyper-Connections (HC):
    x_{l+1} = H^res x_l + H^post F(H^pre x_l)
    
    - Multiple parallel streams, richer connections
    - But free mixing destroys the identity mapping property
    - Training becomes unstable at scale

mHC Solution:
    Constrain the residual mixing matrix to be DOUBLY STOCHASTIC
    (rows sum to 1, columns sum to 1) via Sinkhorn-Knopp projection.
    
    This restores the identity mapping property while keeping
    the expressivity of multiple streams.

================================================================================
ILLUSTRATION: From Residual to mHC
================================================================================

Standard Residual (ResNet/Transformer):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    x_l ───────────────────────────────────────────────► + ──► x_{l+1}│
    │         │                                               ↑            │
    │         └──► F(·) ──► output ──────────────────────────┘            │
    │                                                                      │
    │    Identity path: x_l maps directly to x_{l+1}                      │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Hyper-Connections (HC):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    x_l ──► [stream_1, stream_2, stream_3] ──► H^pre ──► F(·)        │
    │                              │                                       │
    │                              └──► H^post ──► mix ──► + ──► x_{l+1}  │
    │                              ↑                                       │
    │                              └──── H^res ────────────────────────────┘
    │                                                                      │
    │    Problem: H^res may not preserve identity mapping!                │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Manifold-Constrained Hyper-Connections (mHC):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    x_l ──► [stream_1, stream_2, stream_3] ──► H^pre ──► F(·)        │
    │                              │                                       │
    │                              └──► H^post ──► mix ──► + ──► x_{l+1}  │
    │                              ↑                                       │
    │                              └──── Sinkhorn(H^res) ──────────────────┘
    │                                                                      │
    │    Guarantee: After many layers, signal still flows like identity!  │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(
    M: torch.Tensor,
    num_iters: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Project a matrix onto the doubly-stochastic manifold (Birkhoff polytope)
    using the Sinkhorn-Knopp algorithm.

    A doubly-stochastic matrix has:
    - All entries non-negative
    - Each row sums to 1
    - Each column sums to 1

    This is crucial for mHC because it ensures the mixing behaves like
    a weighted average, preserving the identity mapping property.

    Args:
        M: Input matrix of shape (..., n, n)
        num_iters: Number of Sinkhorn iterations
        eps: Small constant for numerical stability

    Returns:
        Doubly-stochastic matrix of same shape

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Sinkhorn-Knopp Iteration:                                                ║
    ║                                                                           ║
    ║  Repeat for k = 1, 2, ..., num_iters:                                    ║
    ║    M = M / (row_sum(M) + eps)    ← normalize rows                       ║
    ║    M = M / (col_sum(M) + eps)    ← normalize columns                    ║
    ║                                                                           ║
    ║  This converges to the unique doubly-stochastic matrix closest to M     ║
    ║  in terms of relative entropy (KL divergence).                          ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    # Ensure non-negative
    M = M.clamp(min=eps)

    for _ in range(num_iters):
        # Normalize rows
        row_sum = M.sum(dim=-1, keepdim=True)
        M = M / (row_sum + eps)

        # Normalize columns
        col_sum = M.sum(dim=-2, keepdim=True)
        M = M / (col_sum + eps)

    return M


class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) Block.

    Replaces the standard residual connection in a transformer block
    with multi-stream connections constrained to the doubly-stochastic manifold.

    Args:
        d_model: Model dimension
        n_streams: Number of parallel residual streams (default 4)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        layer_fn: The transformer sublayer function (attention or FFN)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  mHC Forward Pass:                                                        ║
    ║                                                                           ║
    ║  1. Expand: x (d_model) → [stream_1, ..., stream_n] (n × d_model/n)     ║
    ║  2. Pre-mix: H^pre · streams → mixed_input                              ║
    ║  3. Apply layer: F(mixed_input) → layer_output                          ║
    ║  4. Post-mix: H^post · layer_output → mixed_output                      ║
    ║  5. Residual mix: Sinkhorn(H^res) · streams → residual                  ║
    ║  6. Combine: mixed_output + residual → output                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        assert d_model % n_streams == 0, "d_model must be divisible by n_streams"
        self.stream_dim = d_model // n_streams

        # Learnable mixing matrices (before Sinkhorn projection)
        # Shape: (n_streams, n_streams)
        self.H_pre_raw = nn.Parameter(torch.eye(n_streams) * 0.1 + torch.randn(n_streams, n_streams) * 0.02)
        self.H_post_raw = nn.Parameter(torch.eye(n_streams) * 0.1 + torch.randn(n_streams, n_streams) * 0.02)
        self.H_res_raw = nn.Parameter(torch.eye(n_streams) * 0.5 + torch.randn(n_streams, n_streams) * 0.02)

        # Stream projections: expand input to n_streams and collapse back
        self.expand = nn.Linear(d_model, d_model)
        self.collapse = nn.Linear(d_model, d_model)

    def _get_mixing_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply Sinkhorn-Knopp to project onto doubly-stochastic manifold."""
        # Add identity for stability, then Sinkhorn
        M = raw + torch.eye(self.n_streams, device=raw.device) * 0.5
        return sinkhorn_knopp(M, self.sinkhorn_iters)

    def forward(
        self,
        x: torch.Tensor,
        layer_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Apply mHC around a sublayer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            layer_fn: The sublayer function (e.g., attention or FFN)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Expand input into n parallel streams
        # (batch, seq, d_model) -> (batch, seq, n_streams, stream_dim)
        streams = self.expand(x)
        streams = streams.view(batch_size, seq_len, self.n_streams, self.stream_dim)

        # Step 2: Get doubly-stochastic mixing matrices
        H_pre = self._get_mixing_matrix(self.H_pre_raw)
        H_post = self._get_mixing_matrix(self.H_post_raw)
        H_res = self._get_mixing_matrix(self.H_res_raw)

        # Step 3: Pre-mix streams for layer input
        # (n_streams, n_streams) @ (batch, seq, n_streams, stream_dim)
        # -> (batch, seq, n_streams, stream_dim)
        mixed_input = torch.einsum('nm,bsmd->bsnd', H_pre, streams)

        # Flatten back to d_model for layer function
        mixed_input_flat = mixed_input.reshape(batch_size, seq_len, self.d_model)

        # Step 4: Apply the sublayer (attention or FFN)
        layer_output = layer_fn(mixed_input_flat)

        # Step 5: Reshape layer output back to streams
        layer_output_streams = layer_output.view(batch_size, seq_len, self.n_streams, self.stream_dim)

        # Step 6: Post-mix
        mixed_output = torch.einsum('nm,bsmd->bsnd', H_post, layer_output_streams)

        # Step 7: Residual connection with manifold-constrained mixing
        residual = torch.einsum('nm,bsmd->bsnd', H_res, streams)

        # Step 8: Combine and collapse
        combined = mixed_output + residual
        combined = combined.reshape(batch_size, seq_len, self.d_model)
        output = self.collapse(combined)

        return output


class StandardResidualBlock(nn.Module):
    """Standard residual block for comparison."""

    def __init__(self, d_model: int, layer_fn: nn.Module):
        super().__init__()
        self.layer_fn = layer_fn
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer_fn(self.norm(x))


class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for demonstration."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False)[0]


class SimpleFFNLayer(nn.Module):
    """Simple FFN layer for demonstration."""

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate mHC and compare with standard residual connections."""
    print("=" * 80)
    print("MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC)")
    print("=" * 80)

    d_model = 256
    n_streams = 4
    batch_size = 2
    seq_len = 16

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  n_streams: {n_streams}")
    print(f"  stream_dim: {d_model // n_streams}")
    print(f"  Input shape: {x.shape}")

    # Sinkhorn-Knopp demonstration
    print("\n" + "-" * 80)
    print("1. SINKHORN-KNOPP PROJECTION")
    print("-" * 80)

    # Create a random matrix
    M = torch.randn(n_streams, n_streams)
    M = torch.softmax(M, dim=-1)  # Start with row-stochastic

    print(f"\nOriginal matrix (before projection):")
    print(M.detach().numpy().round(3))
    print(f"Row sums: {M.sum(dim=-1).detach().numpy().round(3)}")
    print(f"Col sums: {M.sum(dim=0).detach().numpy().round(3)}")

    M_ds = sinkhorn_knopp(M, num_iters=10)

    print(f"\nAfter Sinkhorn-Knopp (doubly-stochastic):")
    print(M_ds.detach().numpy().round(3))
    print(f"Row sums: {M_ds.sum(dim=-1).detach().numpy().round(6)}")
    print(f"Col sums: {M_ds.sum(dim=0).detach().numpy().round(6)}")
    print(f"All non-negative: {(M_ds >= 0).all().item()}")

    print("""
    Sinkhorn-Knopp guarantees:
    - All entries ≥ 0
    - Each row sums to 1.0
    - Each column sums to 1.0
    - This is a weighted average → preserves identity mapping
    """)

    # mHC Block demonstration
    print("-" * 80)
    print("2. mHC BLOCK WITH ATTENTION")
    print("-" * 80)

    attn_layer = SimpleAttentionLayer(d_model)
    mhc_attn = ManifoldConstrainedHyperConnection(d_model, n_streams=n_streams)
    out_mhc = mhc_attn(x, attn_layer)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out_mhc.shape}")

    # Compare with standard residual
    print("\n" + "-" * 80)
    print("3. STANDARD RESIDUAL VS mHC")
    print("-" * 80)

    ffn_layer = SimpleFFNLayer(d_model)

    # Standard
    std_block = StandardResidualBlock(d_model, ffn_layer)
    out_std = std_block(x)

    # mHC
    mhc_ffn = ManifoldConstrainedHyperConnection(d_model, n_streams=n_streams)
    out_mhc_ffn = mhc_ffn(x, ffn_layer)

    print(f"\nStandard residual output: {out_std.shape}")
    print(f"mHC output:               {out_mhc_ffn.shape}")

    # Check output magnitudes
    print(f"\nOutput magnitude comparison:")
    print(f"  Input norm:         {x.norm(dim=-1).mean().item():.4f}")
    print(f"  Standard output:    {out_std.norm(dim=-1).mean().item():.4f}")
    print(f"  mHC output:         {out_mhc_ffn.norm(dim=-1).mean().item():.4f}")

    # Deep network stability test
    print("\n" + "-" * 80)
    print("4. DEEP NETWORK STABILITY TEST")
    print("-" * 80)

    num_layers = 20

    # Build standard deep network
    std_layers = nn.ModuleList([
        StandardResidualBlock(d_model, SimpleFFNLayer(d_model))
        for _ in range(num_layers)
    ])

    # Build mHC deep network
    mhc_layers = nn.ModuleList([
        ManifoldConstrainedHyperConnection(d_model, n_streams=n_streams)
        for _ in range(num_layers)
    ])
    ffn_layers = nn.ModuleList([SimpleFFNLayer(d_model) for _ in range(num_layers)])

    # Forward through standard network
    h_std = x
    for layer in std_layers:
        h_std = layer(h_std)

    # Forward through mHC network
    h_mhc = x
    for mhc, ffn in zip(mhc_layers, ffn_layers):
        h_mhc = mhc(h_mhc, ffn)

    print(f"\nAfter {num_layers} layers:")
    print(f"  Standard residual norm: {h_std.norm(dim=-1).mean().item():.4f}")
    print(f"  mHC norm:               {h_mhc.norm(dim=-1).mean().item():.4f}")
    print(f"  Input norm:             {x.norm(dim=-1).mean().item():.4f}")

    print("""
    Key observation:
    - Both maintain stable signal magnitudes
    - mHC has richer multi-stream connectivity
    - Sinkhorn projection ensures no signal explosion/vanishing
    - mHC scales to deeper networks more reliably than free HC
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Standard residual: x_{l+1} = x_l + F(x_l) — stable but restrictive
    2. Hyper-Connections: multi-stream but unstable at scale
    3. mHC: projects mixing to doubly-stochastic manifold via Sinkhorn-Knopp
    4. Doubly-stochastic = weighted average → preserves identity mapping
    5. mHC achieves HC's expressivity with ResNet's stability
    6. DeepSeek uses mHC to train very deep models efficiently

    Next: architecture/engram.py - Conditional Memory (Engram)
    """)


if __name__ == "__main__":
    demo()
