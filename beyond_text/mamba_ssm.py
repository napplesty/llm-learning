"""
================================================================================
MAMBA / STATE SPACE MODELS (SSM)
================================================================================

Mamba is a new architecture that challenges Transformers by offering:
- Linear time complexity O(N) vs Transformer's O(N²)
- Fast inference (5× higher throughput)
- Ability to handle very long sequences (up to millions of tokens)

Key Innovation: Selective State Space Models
- Input-dependent parameters (unlike traditional SSMs)
- Hardware-aware parallel algorithm
- Combines benefits of RNNs (fast inference) and Transformers (parallel training)

================================================================================
ILLUSTRATION: Transformer vs Mamba Complexity
================================================================================

    Transformer Attention:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Sequence Length: N                                                │
    │    Memory: O(N²)        ← Must store N×N attention matrix           │
    │    Compute: O(N²)       ← Compare every token with every token      │
    │                                                                      │
    │    Example: N=1000 → 1M operations                                  │
    │             N=10000 → 100M operations (100× more!)                  │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Mamba SSM:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Sequence Length: N                                                │
    │    Memory: O(N)         ← Only store compressed state               │
    │    Compute: O(N)        ← Process each token once                   │
    │                                                                      │
    │    Example: N=1000 → 1K operations                                  │
    │             N=10000 → 10K operations (10× more, not 100×!)          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: State Space Model Basics
================================================================================

    Traditional State Space Model (SSM):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    State Equation:   h'(t) = A·h(t) + B·x(t)                        │
    │    Output Equation:  y(t)  = C·h(t) + D·x(t)                        │
    │                                                                      │
    │    Where:                                                            │
    │      x(t) = input signal                                            │
    │      h(t) = hidden state (compressed memory)                        │
    │      y(t) = output                                                  │
    │      A, B, C, D = learnable matrices                                │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Discretization (for sequence data):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    h(t) = Ā·h(t-1) + B̄·x(t)                                         │
    │    y(t) = C·h(t)                                                    │
    │                                                                      │
    │    Ā = exp(Δ·A)  ← Discretized A using step size Δ                  │
    │    B̄ = (Δ·A)⁻¹·(exp(Δ·A) - I)·Δ·B                                   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Three Representations:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    1. Continuous: For continuous signals                            │
    │       h'(t) = Ah(t) + Bx(t),  y(t) = Ch(t)                          │
    │                                                                      │
    │    2. Recurrent: For fast inference (like RNN)                      │
    │       hₖ = Āhₖ₋₁ + B̄xₖ,  yₖ = Chₖ                                   │
    │                                                                      │
    │    3. Convolutional: For parallel training (like CNN)               │
    │       y = K * x  where K is SSM kernel                              │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Mamba's Key Innovation - Selective SSM
================================================================================

    Problem with Traditional SSMs:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Matrices A, B, C are FIXED (Linear Time Invariant)               │
    │                                                                      │
    │    → Cannot do content-based reasoning                              │
    │    → Cannot selectively remember or forget information              │
    │    → Poor performance on language tasks                             │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Mamba's Solution - Input-Dependent Parameters:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    B = Linear(x)    ← B depends on input!                           │
    │    C = Linear(x)    ← C depends on input!                           │
    │    Δ = Linear(x)    ← Step size depends on input!                   │
    │                                                                      │
    │    Why this matters:                                                │
    │    - Small Δ → Ignore current input, use context more               │
    │    - Large Δ → Focus on current input, update state heavily         │
    │                                                                      │
    │    Result: Model can selectively propagate or forget information    │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Selective Scan Algorithm:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Input: x [B, L, D]                                               │
    │           ↓                                                          │
    │    For each position t:                                             │
    │        Bₜ, Cₜ, Δₜ = projection(xₜ)   ← Input-dependent              │
    │        Āₜ = exp(Δₜ·A)                                              │
    │        hₜ = Āₜ·hₜ₋₁ + B̄ₜ·xₜ           ← Update state                │
    │        yₜ = Cₜ·hₜ                      ← Compute output             │
    │           ↓                                                          │
    │    Output: y [B, L, D]                                              │
    │                                                                      │
    │    Note: Parallelized using associative scan!                       │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Mamba Block Architecture
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          Mamba Block                                     │
    │                                                                          │
    │    Input x [B, L, D]                                                    │
    │         │                                                                │
    │         ▼                                                                │
    │    ┌─────────────────┐                                                  │
    │    │    RMSNorm      │                                                  │
    │    └─────────────────┘                                                  │
    │         │                                                                │
    │         ▼                                                                │
    │    ┌─────────────────┐                                                  │
    │    │  Linear (2×D)   │  ← Project to 2× dimension                      │
    │    └─────────────────┘                                                  │
    │         │                                                                │
    │         ├──────────────────────┐                                        │
    │         │                      │                                        │
    │         ▼                      ▼                                        │
    │    ┌─────────────────┐   ┌─────────────────┐                           │
    │    │  Conv1D (causal)│   │     SiLU        │  ← Gating                │
    │    └─────────────────┘   └─────────────────┘                           │
    │         │                      │                                        │
    │         ▼                      │                                        │
    │    ┌─────────────────┐         │                                        │
    │    │  SilU           │         │                                        │
    │    └─────────────────┘         │                                        │
    │         │                      │                                        │
    │         ▼                      │                                        │
    │    ┌─────────────────────────┐ │                                        │
    │    │    Selective SSM        │ │                                        │
    │    │  - x_proj → (Δ, B, C)   │ │                                        │
    │    │  - dt_proj → Δ          │ │                                        │
    │    │  - selective_scan       │ │                                        │
    │    │  - D skip connection    │ │                                        │
    │    └─────────────────────────┘ │                                        │
    │         │                      │                                        │
    │         ▼                      ▼                                        │
    │         └─────────×────────────┘  ← Element-wise multiplication        │
    │                  │                                                     │
    │                  ▼                                                     │
    │         ┌─────────────────┐                                            │
    │         │  Linear (D)     │                                            │
    │         └─────────────────┘                                            │
    │                  │                                                     │
    │                  ▼                                                     │
    │         Output + Residual                                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# 1. State Space Model Configuration
# =============================================================================

@dataclass
class SSMConfig:
    """Configuration for State Space Model."""
    d_model: int = 256           # Model dimension
    d_state: int = 16            # SSM state dimension (N)
    d_conv: int = 4              # Local convolution width
    expand: int = 2              # Block expansion factor
    dt_rank: str = "auto"        # Rank of Δ projection
    dt_min: float = 0.001        # Minimum Δ
    dt_max: float = 0.1          # Maximum Δ
    dt_init: str = "random"      # Δ initialization
    dt_scale: float = 1.0        # Δ scale
    conv_bias: bool = True       # Use bias in conv
    bias: bool = False           # Use bias in linear layers


# =============================================================================
# 2. RMSNorm (from earlier modules)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# 3. Selective Scan - Core SSM Operation (Reference Implementation)
# =============================================================================

def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Reference implementation of selective scan.
    
    This is the core operation of Mamba - the selective state space model.
    
    Args:
        u: Input tensor [B, D, L]
        delta: Step size Δ [B, D, L]
        A: State transition matrix [D, N]
        B: Input matrix [B, N, L]
        C: Output matrix [B, N, L]
        D: Skip connection [D]
        delta_bias: Bias for delta [D]
        delta_softplus: Apply softplus to delta
    
    Returns:
        Output tensor [B, D, L]
    
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Selective Scan Algorithm:                                                 ║
    ║                                                                           ║
    ║  For each time step t:                                                    ║
    ║    1. Discretize: Ā = exp(Δ·A),  B̄ = Δ·B                                 ║
    ║    2. Update state: hₜ = Ā·hₜ₋₁ + B̄·xₜ                                   ║
    ║    3. Compute output: yₜ = C·hₜ + D·xₜ                                    ║
    ║                                                                           ║
    ║  This is O(L) serially, but can be parallelized with associative scan!    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    
    if delta_softplus:
        delta = F.softplus(delta)
    
    batch, dim, L = u.shape
    d_state = A.shape[1]
    
    # Initialize state
    x = torch.zeros((batch, dim, d_state), device=u.device, dtype=u.float())
    
    # Precompute discretized A
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    
    # Compute discretized B * u
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    
    # Sequential scan (slow but clear)
    # In practice, this is parallelized using associative scan
    outputs = []
    for i in range(L):
        # Update state: h_t = A_bar * h_{t-1} + B_bar * x_t
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        
        # Compute output: y_t = C * h_t
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        outputs.append(y)
    
    # Stack outputs
    y = torch.stack(outputs, dim=2)  # [B, D, L]
    
    # Add skip connection
    if D is not None:
        y = y + u * rearrange(D, "d -> d 1")
    
    return y.to(dtype_in)


# =============================================================================
# 4. Mamba Block
# =============================================================================

class MambaBlock(nn.Module):
    """
    Mamba Block with Selective State Space Model.
    
    This is a simplified implementation for educational purposes.
    Production code uses optimized CUDA kernels.
    
    Args:
        config: SSMConfig with model parameters
    
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Mamba Block Components:                                                   ║
    ║                                                                           ║
    ║  1. Input Projection: Linear layer to expand dimension                    ║
    ║  2. Causal Conv1D: Local convolution for nearby context                   ║
    ║  3. Selective SSM: Core state space model with input-dependent params     ║
    ║  4. Output Projection: Linear layer back to original dimension            ║
    ║                                                                           ║
    ║  Key insight: The SSM parameters (B, C, Δ) depend on the input,           ║
    ║  allowing the model to selectively remember or forget information.        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if config.dt_rank == "auto" else config.dt_rank
        
        # Input projection: project to 2x dimension for gating
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=config.conv_bias
        )
        
        # Activation
        self.act = nn.SiLU()
        
        # SSM projections
        # Projects input to (Δ_rank, d_state, d_state) for Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # Projects Δ_rank to d_inner for Δ
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize Δ projection
        dt_init_std = self.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize Δ bias to be in [dt_min, dt_max] after softplus
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # SSM parameter A (state transition)
        # Using S4D initialization: A = -exp(A_log) where A_log is learnable
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # SSM parameter D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor [B, L, D]
        
        Returns:
            Output tensor [B, L, D]
        """
        batch, seqlen, dim = x.shape
        
        # Input projection with gating
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Causal convolution
        # Rearrange for conv1d: [B, L, D] -> [B, D, L]
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]  # Remove padding
        x_conv = x_conv.transpose(1, 2)  # [B, L, D]
        x_conv = self.act(x_conv)
        
        # SSM parameters from input
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        
        # Split into Δ, B, C
        delta, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Project Δ
        delta = self.dt_proj(delta)  # [B, L, d_inner]
        
        # Get A from log space
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Rearrange for selective scan
        u = x_conv.transpose(1, 2)  # [B, d_inner, L]
        delta = delta.transpose(1, 2)  # [B, d_inner, L]
        B = B.transpose(1, 2)  # [B, d_state, L]
        C = C.transpose(1, 2)  # [B, d_state, L]
        
        # Selective scan
        y = selective_scan_ref(
            u, delta, A, B, C, 
            D=self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True
        )
        
        # Rearrange back
        y = y.transpose(1, 2)  # [B, L, d_inner]
        
        # Gating
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


# =============================================================================
# 5. Complete Mamba Model
# =============================================================================

class MambaModel(nn.Module):
    """
    Complete Mamba Language Model.
    
    This replaces Transformer layers with Mamba blocks,
    achieving O(N) complexity instead of O(N²).
    
    Args:
        vocab_size: Vocabulary size
        config: SSMConfig with model parameters
        num_layers: Number of Mamba layers
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        config: Optional[SSMConfig] = None,
        num_layers: int = 12
    ):
        super().__init__()
        
        self.config = config or SSMConfig()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, self.config.d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(self.config) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm_f = RMSNorm(self.config.d_model)
        
        # Output head
        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [B, L]
        
        Returns:
            Logits [B, L, vocab_size]
        """
        # Embedding
        x = self.embedding(input_ids)
        
        # Mamba layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        # Final norm
        x = self.norm_f(x)
        
        # Output
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 6. Hybrid Mamba-Transformer Model
# =============================================================================

class HybridBlock(nn.Module):
    """
    Hybrid block combining Mamba and Attention.
    
    Modern architectures like Jamba alternate between Mamba and Attention
    layers to get benefits of both:
    - Mamba: Fast inference, long context
    - Attention: Content-based reasoning, proven performance
    
    Args:
        d_model: Model dimension
        use_attention: If True, use attention; otherwise use Mamba
    """
    
    def __init__(self, d_model: int = 256, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        
        if use_attention:
            # Simplified attention block
            self.norm = RMSNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        else:
            # Mamba block
            self.mamba = MambaBlock(SSMConfig(d_model=d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            # Attention
            residual = x
            x = self.norm(x)
            
            B, L, D = x.shape
            
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Simplified single-head attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            
            # Causal mask
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = self.out_proj(out)
            
            return residual + out
        else:
            # Mamba
            return x + self.mamba(x)


# =============================================================================
# 7. Demonstration
# =============================================================================

def demo_mamba():
    """Demonstrate Mamba architecture."""
    print("=" * 70)
    print("MAMBA / STATE SPACE MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    config = SSMConfig(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    print(f"\nMamba Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_state (N): {config.d_state}")
    print(f"  d_conv: {config.d_conv}")
    print(f"  expand: {config.expand}")
    print(f"  d_inner: {config.d_model * config.expand}")
    
    # Create Mamba block
    mamba = MambaBlock(config)
    
    # Test input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = mamba(x)
    print(f"Output shape: {output.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in mamba.parameters())
    print(f"Mamba block parameters: {params:,}")
    
    # Create full model
    print("\n" + "-" * 70)
    print("Full Mamba Model")
    print("-" * 70)
    
    model = MambaModel(
        vocab_size=10000,
        config=config,
        num_layers=6
    )
    
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    # Test generation
    input_ids = torch.randint(0, 10000, (1, 32))
    logits = model(input_ids)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Compare with Transformer complexity
    print("\n" + "-" * 70)
    print("Complexity Comparison")
    print("-" * 70)
    
    for seq_len in [128, 512, 2048, 8192]:
        transformer_ops = seq_len ** 2
        mamba_ops = seq_len
        print(f"Sequence length {seq_len:5d}: "
              f"Transformer O(N²) = {transformer_ops:10,}, "
              f"Mamba O(N) = {mamba_ops:5,}, "
              f"Speedup = {transformer_ops/mamba_ops:8.1f}x")
    
    print("\n" + "=" * 70)
    print("Mamba offers linear scaling with sequence length!")
    print("This enables processing much longer sequences efficiently.")
    print("=" * 70)


def demo_hybrid():
    """Demonstrate Hybrid Mamba-Transformer."""
    print("\n" + "=" * 70)
    print("HYBRID MAMBA-TRANSFORMER DEMONSTRATION")
    print("=" * 70)
    
    d_model = 128
    
    # Create hybrid model: 4 Mamba layers, 2 Attention layers
    layers = nn.ModuleList([
        HybridBlock(d_model, use_attention=(i % 3 == 2))  # Every 3rd is attention
        for i in range(6)
    ])
    
    print("\nHybrid Architecture (6 layers):")
    for i, layer in enumerate(layers):
        layer_type = "Attention" if layer.use_attention else "Mamba"
        print(f"  Layer {i}: {layer_type}")
    
    # Test
    x = torch.randn(2, 64, d_model)
    for layer in layers:
        x = layer(x)
    
    print(f"\nInput shape: {torch.randn(2, 64, d_model).shape}")
    print(f"Output shape: {x.shape}")
    
    print("\nHybrid models like Jamba achieve:")
    print("  - 3× higher throughput than pure Transformers")
    print("  - 256K+ context length")
    print("  - Best of both worlds!")


# =============================================================================
# Helper for rearrange (simplified version without einops)
# =============================================================================

def rearrange(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
    """
    Simplified rearrange function (subset of einops.rearrange).
    Only supports patterns used in this module.
    """
    if pattern == "d -> d 1":
        return tensor.unsqueeze(-1)
    elif pattern == "d -> 1 d":
        return tensor.unsqueeze(0)
    else:
        return tensor


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_mamba()
    demo_hybrid()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Mamba uses Selective State Space Models (S6) instead of Attention

2. Key Innovation: Input-dependent parameters (B, C, Δ)
   - Allows selective remembering/forgetting
   - Enables content-based reasoning

3. Complexity: O(N) vs Transformer's O(N²)
   - Much faster for long sequences
   - Enables processing millions of tokens

4. Hardware Efficiency:
   - Uses parallel scan algorithm
   - Kernel fusion for memory efficiency
   - Recurrent mode for fast inference

5. Hybrid Models (Jamba, etc.):
   - Combine Mamba + Attention
   - Get benefits of both architectures
   - State-of-the-art performance

6. Use Cases:
   - Long document processing
   - DNA/Genomics sequences
   - Audio processing
   - Time series analysis
""")
