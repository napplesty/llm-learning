"""
================================================================================
LLM Learning Module 12: PARAMETER-EFFICIENT FINE-TUNING (PEFT)
================================================================================

Fine-tuning techniques that modify only a small subset of parameters:

1. LoRA (Low-Rank Adaptation) - Factorized weight updates
2. QLoRA - Quantized LoRA for extreme memory efficiency
3. Prefix Tuning - Learn soft prompts
4. Adapter Layers - Small bottleneck modules

================================================================================
ILLUSTRATION: LoRA (Low-Rank Adaptation)
================================================================================

Standard Fine-tuning:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    W (d × d)                                                         │
    │    ↓                                                                 │
    │    y = Wx                                                            │
    │                                                                      │
    │    All d² parameters updated (expensive!)                           │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

LoRA Fine-tuning:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    W (d × d) ─────────────────────────────────┐                      │
    │    ↓                                           │                      │
    │    Wx                                          │                      │
    │         ╭─────────────────────────────────────┤                      │
    │         │                                     │                      │
    │    A (r × d)                                  │                      │
    │         ↓                                     │                      │
    │    B (d × r)                                  │ +                    │
    │         ↓                                     │                      │
    │    BAx ───────────────────────────────────────┘                      │
    │                                                                      │
    │    y = Wx + BAx = (W + BA)x                                          │
    │                                                                      │
    │    Only 2 × d × r parameters updated (r << d)                       │
    │                                                                      │
    │    Example: d=4096, r=8                                              │
    │    Full: 4096² = 16,777,216 params                                   │
    │    LoRA: 2 × 4096 × 8 = 65,536 params (256x fewer!)                 │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# 1. LoRA (Low-Rank Adaptation)
# =============================================================================

@dataclass
class LoRAConfig:
    """Configuration for LoRA."""
    r: int = 8                    # Rank of low-rank matrices
    alpha: float = 16.0           # Scaling factor
    dropout: float = 0.0          # Dropout probability
    target_modules: List[str] = None  # Modules to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Instead of updating the full weight matrix W, LoRA adds a low-rank
    decomposition BA where B is (out_features, r) and A is (r, in_features).

    Formula: y = Wx + (alpha/r) × BAx

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Why does LoRA work?                                                      ║
    ║                                                                           ║
    ║  The hypothesis is that model adaptation has low "intrinsic dimension"    ║
    ║  - meaning the changes needed can be captured in a low-rank subspace.     ║
    ║                                                                           ║
    ║  Empirically, r=4 to r=8 often works as well as full fine-tuning!        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original weight (frozen during LoRA training)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if True else None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_normal_(self.lora_B, a=math.sqrt(5))

        self.merged = False
        self.merge_weights = merge_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional LoRA."""
        # Original transformation
        result = F.linear(x, self.weight, self.bias)

        # Add LoRA if not merged
        if not self.merged:
            # y = Wx + scaling × B × A × x
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            # W' = W + scaling × BA
            delta_W = self.lora_B @ self.lora_A * self.scaling
            self.weight.data += delta_W
            self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights."""
        if self.merged:
            delta_W = self.lora_B @ self.lora_A * self.scaling
            self.weight.data -= delta_W
            self.merged = False


class LoRAModel(nn.Module):
    """
    Wrapper to apply LoRA to specific modules of a model.

    Args:
        model: Base model to apply LoRA to
        config: LoRA configuration

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  LoRA Application Strategy:                                               ║
    ║                                                                           ║
    ║  Common target modules:                                                   ║
    ║  - Attention: q_proj, k_proj, v_proj, o_proj                             ║
    ║  - FFN: gate_proj, up_proj, down_proj                                    ║
    ║                                                                           ║
    ║  Best practices:                                                          ║
    ║  - Start with q_proj, v_proj only (minimal)                              ║
    ║  - Increase r if underfitting                                            ║
    ║  - Apply to all linear layers for maximum capacity                       ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.lora_modules: Dict[str, LoRALinear] = {}

        # Find and replace target modules
        self._apply_lora()

    def _apply_lora(self):
        """Replace target modules with LoRA versions."""
        for name, module in list(self.model.named_modules()):
            # Check if this module should have LoRA
            for target in self.config.target_modules:
                if target in name and isinstance(module, nn.Linear):
                    # Create LoRA replacement
                    lora_module = LoRALinear(
                        module.in_features,
                        module.out_features,
                        self.config.r,
                        self.config.alpha,
                        self.config.dropout,
                    )
                    # Copy original weights
                    lora_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_module.bias.data = module.bias.data.clone()

                    # Replace in model
                    self._replace_module(name, lora_module)
                    self.lora_modules[name] = lora_module

    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model hierarchy."""
        parts = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get only LoRA parameters for optimizer."""
        params = []
        for module in self.lora_modules.values():
            params.extend([module.lora_A, module.lora_B])
        return params

    def merge_and_save(self) -> nn.Module:
        """Merge LoRA weights and return model for deployment."""
        for module in self.lora_modules.values():
            module.merge_weights()
        return self.model

    def save_lora_weights(self, path: str):
        """Save only LoRA weights (small file)."""
        state_dict = {}
        for name, module in self.lora_modules.items():
            state_dict[f"{name}.lora_A"] = module.lora_A.data
            state_dict[f"{name}.lora_B"] = module.lora_B.data
        torch.save(state_dict, path)

    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        state_dict = torch.load(path)
        for name, module in self.lora_modules.items():
            module.lora_A.data = state_dict[f"{name}.lora_A"]
            module.lora_B.data = state_dict[f"{name}.lora_B"]


# =============================================================================
# 2. QLoRA (Quantized LoRA)
# =============================================================================

class QuantizedLinear(nn.Module):
    """
    4-bit NormalFloat Quantized Linear Layer.

    QLoRA quantizes the base model weights to 4-bit NF4 format,
    dramatically reducing memory usage while maintaining performance.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  NF4 (NormalFloat 4-bit):                                                 ║
    ║                                                                           ║
    ║  - 4 bits per weight (16 quantization levels)                            ║
    ║  - Levels distributed normally (more precision near 0)                   ║
    ║  - Double quantization for quantization constants                        ║
    ║                                                                           ║
    ║  Memory comparison for 7B model:                                          ║
    ║  - FP16: 14 GB                                                            ║
    ║  - 8-bit: 7 GB                                                            ║
    ║  - 4-bit (NF4): 3.5 GB                                                    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight storage
        # In real implementation, uses bitsandbytes library
        self.weight_quantized = nn.Parameter(
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(out_features),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize(self, weight: torch.Tensor):
        """Quantize FP16 weights to NF4 (simplified)."""
        # Simplified quantization - real NF4 uses learned quantiles
        abs_max = weight.abs().max()
        self.weight_scale.data = abs_max

        # Quantize to 4-bit (simplified to int8 for demo)
        normalized = weight / (abs_max + 1e-8)
        quantized = (normalized * 127).round().clamp(-128, 127).to(torch.int8)
        self.weight_quantized.data = quantized.view(torch.uint8) // 2 + 128

    def dequantize(self) -> torch.Tensor:
        """Dequantize NF4 to FP16."""
        # Simplified dequantization
        quantized = (self.weight_quantized.data.view(torch.int8) - 128) * 2
        return quantized.float() * self.weight_scale / 127

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dequantization."""
        weight = self.dequantize()
        return F.linear(x, weight, self.bias)


class QLoRAModel(nn.Module):
    """
    QLoRA: Quantized base model + LoRA adapters.

    Combines 4-bit quantization with LoRA for extreme memory efficiency.

    Args:
        model: Base model to quantize
        lora_config: LoRA configuration
    """

    def __init__(self, model: nn.Module, lora_config: LoRAConfig):
        super().__init__()
        self.lora_config = lora_config

        # Quantize base model (simplified)
        self.quantized_layers = {}
        self.lora_layers = {}

        # In practice, use bitsandbytes for actual quantization
        print("Quantizing base model to 4-bit...")
        # ... quantization code ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through quantized model with LoRA.
        
        This is a simplified demonstration. In practice, you would:
        1. Pass input through each quantized layer
        2. Add LoRA adapter outputs where applicable
        3. Handle the full model architecture
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor after processing through quantized model
        """
        # In a real implementation, this would process through all layers
        # For demonstration, we show the concept
        
        # Process through quantized layers (simplified)
        # In practice, use the actual model architecture
        for name, layer in self.quantized_layers.items():
            x = layer(x)
            
            # Add LoRA if this layer has an adapter
            if name in self.lora_layers:
                lora_out = self.lora_layers[name](x)
                x = x + lora_out
        
        return x
    
    def add_lora_adapter(self, layer_name: str, in_features: int, out_features: int):
        """Add a LoRA adapter to a specific layer."""
        self.lora_layers[layer_name] = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=self.lora_config.r,
            alpha=self.lora_config.alpha,
            dropout=self.lora_config.dropout
        )
    
    def quantize_layer(self, layer_name: str, layer: nn.Module):
        """Quantize a linear layer to 4-bit."""
        if isinstance(layer, nn.Linear):
            quantized = QuantizedLinear(layer.in_features, layer.out_features)
            quantized.quantize(layer.weight.data)
            if layer.bias is not None:
                quantized.bias = layer.bias
            self.quantized_layers[layer_name] = quantized
        else:
            # Keep non-linear layers as-is
            self.quantized_layers[layer_name] = layer


# =============================================================================
# 3. Prefix Tuning
# =============================================================================

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Add learnable soft prompts to the input.

    Instead of modifying model weights, prefix tuning prepends a sequence
    of learnable "virtual tokens" that guide the model's behavior.

    Args:
        num_layers: Number of transformer layers
        d_model: Model dimension
        prefix_len: Length of learnable prefix
        num_heads: Number of attention heads

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Prefix Tuning Visualization:                                              ║
    ║                                                                           ║
    ║  Standard input:                                                          ║
    ║    [T1, T2, T3, T4, T5]                                                   ║
    ║                                                                           ║
    ║  With prefix tuning:                                                      ║
    ║    [P1, P2, P3, P4, P5, T1, T2, T3, T4, T5]                              ║
    ║     └────── prefix ──────┘ └─── actual input ────┘                       ║
    ║                                                                           ║
    ║  The prefix tokens are:                                                   ║
    ║  - Not actual tokens, but learned embeddings                              ║
    ║  - Injected at every layer (not just input)                              ║
    ║  - Optimized via reparameterization for stability                         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        prefix_len: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        self.d_model = d_model

        # Learnable prefix embeddings for each layer
        # Shape: (num_layers, prefix_len, d_model)
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers, prefix_len, d_model) * 0.02
        )

        # Reparameterization MLP (optional, for stability)
        self.reparam = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def get_prefix(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """Get prefix embeddings for a specific layer."""
        prefix = self.prefix_embeddings[layer_idx]  # (prefix_len, d_model)
        prefix = self.reparam(prefix)  # Apply reparameterization
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, prefix_len, d_model)
        return self.dropout(prefix)

    def prepend_to_input(
        self,
        x: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend prefix to input and return attention mask.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            layer_idx: Current layer index

        Returns:
            concatenated: (batch, prefix_len + seq_len, d_model)
            prefix_mask: Attention mask for prefix positions
        """
        batch_size = x.shape[0]
        prefix = self.get_prefix(layer_idx, batch_size)
        return torch.cat([prefix, x], dim=1)


# =============================================================================
# 4. Adapter Layers
# =============================================================================

class AdapterLayer(nn.Module):
    """
    Bottleneck Adapter Layer.

    Adds a small bottleneck module after each transformer layer.
    Only the adapter weights are trained, base model stays frozen.

    Args:
        d_model: Model dimension
        bottleneck_dim: Bottleneck dimension (typically d_model / 16)
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Adapter Architecture:                                                     ║
    ║                                                                           ║
    ║    Input                                                                  ║
    ║      │                                                                    ║
    ║      ▼                                                                    ║
    ║    ┌─────────────┐                                                        ║
    ║    │ LayerNorm   │                                                        ║
    ║    └─────────────┘                                                        ║
    ║      │                                                                    ║
    ║      ├────────────────────────────────────┐                               ║
    ║      │                                    │                               ║
    ║      ▼                                    │                               ║
    ║    ┌─────────────┐                        │                               ║
    ║    │ Down-project│  d → bottleneck        │                               ║
    ║    └─────────────┘                        │                               ║
    ║      │                                    │                               ║
    ║      ▼                                    │                               ║
    ║    ┌─────────────┐                        │                               ║
    ║    │ Activation  │  (ReLU/GELU)           │                               ║
    ║    └─────────────┘                        │                               ║
    ║      │                                    │                               ║
    ║      ▼                                    │                               ║
    ║    ┌─────────────┐                        │                               ║
    ║    │ Up-project  │  bottleneck → d        │                               ║
    ║    └─────────────┘                        │                               ║
    ║      │                                    │                               ║
    ║      ▼                                    │                               ║
    ║    ┌─────────────┐                        │                               ║
    ║    │  Dropout    │                        │                               ║
    ║    └─────────────┘                        │                               ║
    ║      │                                    │                               ║
    ║      └────────────► [ + ] ◄───────────────┘                               ║
    ║                      │                                                    ║
    ║                      ▼                                                    ║
    ║                   Output                                                  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Initialize up_proj to near-zero for stable start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + x


# =============================================================================
# 5. Comparison and Best Practices
# =============================================================================

def print_comparison():
    """Print comparison of PEFT methods."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                    PEFT METHODS COMPARISON                                      ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  Method         │ Trainable Params │ Memory    │ Speed    │ Quality             ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║  Full Fine-tune │ 100%             │ High      │ Slow     │ ★★★★★               ║
║  LoRA           │ 0.1-1%           │ Medium    │ Fast     │ ★★★★☆               ║
║  QLoRA          │ 0.1-1%           │ Very Low  │ Medium   │ ★★★★☆               ║
║  Prefix Tuning  │ 0.1%             │ Low       │ Fast     │ ★★★☆☆               ║
║  Adapters       │ 1-5%             │ Medium    │ Medium   │ ★★★★☆               ║
║                                                                                 ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  When to use each:                                                              ║
║                                                                                 ║
║  LoRA:                                                                         ║
║  - General-purpose fine-tuning                                                 ║
║  - Multiple task adapters on same base model                                   ║
║  - Quick experimentation                                                       ║
║                                                                                 ║
║  QLoRA:                                                                        ║
║  - Limited GPU memory (<16GB)                                                  ║
║  - Fine-tuning very large models (65B+)                                        ║
║  - Single GPU training                                                         ║
║                                                                                 ║
║  Prefix Tuning:                                                                ║
║  - Task-specific customization                                                 ║
║  - When you can't modify model weights                                         ║
║  - Low-resource deployment                                                     ║
║                                                                                 ║
║  Adapters:                                                                     ║
║  - Multi-task learning                                                         ║
║  - When you need modular task switching                                        ║
║  - Production deployment with multiple tasks                                   ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate PEFT techniques."""
    print("=" * 80)
    print("PARAMETER-EFFICIENT FINE-TUNING (PEFT) DEMONSTRATION")
    print("=" * 80)

    # 1. LoRA
    print("\n" + "-" * 80)
    print("1. LoRA (LOW-RANK ADAPTATION)")
    print("-" * 80)

    d_model = 256
    r = 8
    batch_size = 2
    seq_len = 16

    # Standard linear layer
    standard_linear = nn.Linear(d_model, d_model)
    standard_params = sum(p.numel() for p in standard_linear.parameters())

    # LoRA linear layer
    lora_linear = LoRALinear(d_model, d_model, r=r, alpha=16.0)
    lora_params = sum(p.numel() for p in [lora_linear.lora_A, lora_linear.lora_B])

    print(f"\nDimension: {d_model}, Rank: {r}")
    print(f"Standard Linear params: {standard_params:,}")
    print(f"LoRA params (A + B):    {lora_params:,}")
    print(f"Reduction:              {standard_params / lora_params:.1f}x")

    # Forward pass comparison
    x = torch.randn(batch_size, seq_len, d_model)
    standard_out = standard_linear(x)
    lora_out = lora_linear(x)

    print(f"\nStandard output shape: {standard_out.shape}")
    print(f"LoRA output shape:      {lora_out.shape}")

    # 2. LoRA Configuration
    print("\n" + "-" * 80)
    print("2. LoRA CONFIGURATION BEST PRACTICES")
    print("-" * 80)
    print("""
    Recommended LoRA configurations:

    ┌─────────────────┬───────────────┬───────────────┬─────────────────────┐
    │ Model Size      │ Rank (r)      │ Alpha         │ Target Modules      │
    ├─────────────────┼───────────────┼───────────────┼─────────────────────┤
    │ 7B              │ 8-16          │ 16-32         │ q_proj, v_proj      │
    │ 13B             │ 16-32         │ 32            │ q_proj, v_proj      │
    │ 70B             │ 32-64         │ 64            │ all linear layers   │
    └─────────────────┴───────────────┴───────────────┴─────────────────────┘

    Tips:
    - Start with r=8, increase if underfitting
    - Alpha = 2 × r is a good default
    - Apply to q_proj, v_proj first, expand if needed
    - Learning rate: 1e-4 to 5e-4
    """)

    # 3. Prefix Tuning
    print("-" * 80)
    print("3. PREFIX TUNING")
    print("-" * 80)

    prefix_tuning = PrefixTuning(
        num_layers=12,
        d_model=256,
        prefix_len=10,
        num_heads=8,
    )

    prefix_params = sum(p.numel() for p in prefix_tuning.parameters())
    print(f"\nPrefix length: 10")
    print(f"Prefix parameters: {prefix_params:,}")

    # Get prefix for layer 0
    prefix = prefix_tuning.get_prefix(layer_idx=0, batch_size=2)
    print(f"Prefix shape for layer 0: {prefix.shape}")

    # 4. Adapter Layers
    print("\n" + "-" * 80)
    print("4. ADAPTER LAYERS")
    print("-" * 80)

    adapter = AdapterLayer(d_model=256, bottleneck_dim=16)
    adapter_params = sum(p.numel() for p in adapter.parameters())

    x = torch.randn(batch_size, seq_len, 256)
    adapter_out = adapter(x)

    print(f"\nBottleneck dimension: 16")
    print(f"Adapter parameters: {adapter_params:,}")
    print(f"Output shape: {adapter_out.shape}")

    # Comparison
    print("\n" + "-" * 80)
    print("5. METHOD COMPARISON FOR 7B MODEL")
    print("-" * 80)

    # Approximate params for 7B model
    model_params = 7e9
    lora_7b_params = model_params * 0.001  # 0.1%
    adapter_7b_params = model_params * 0.02  # 2%
    prefix_7b_params = model_params * 0.001  # 0.1%

    print(f"\nBase model: 7B parameters")
    print(f"\nTrainable parameters:")
    print(f"  Full fine-tune: {model_params / 1e9:.1f}B (100%)")
    print(f"  LoRA (r=16):    {lora_7b_params / 1e6:.1f}M (0.1%)")
    print(f"  Adapters:       {adapter_7b_params / 1e6:.1f}M (2%)")
    print(f"  Prefix Tuning:  {prefix_7b_params / 1e6:.1f}M (0.1%)")

    print_comparison()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. LoRA: Most popular PEFT method, great balance of efficiency and quality
    2. QLoRA: Enables fine-tuning large models on consumer GPUs
    3. Prefix Tuning: Good for task-specific customization without weight changes
    4. Adapters: Modular approach for multi-task deployment
    5. Always start with small rank (r=8) and increase if needed

    Practical tips:
    - Use LoRA for most fine-tuning tasks
    - Use QLoRA when memory is limited
    - Combine with quantization for maximum efficiency
    - Save only adapter weights (tiny files)

    Next: 13_alignment.py - RLHF, DPO, and Model Alignment
    """)


if __name__ == "__main__":
    demo()
