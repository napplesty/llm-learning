"""
================================================================================
LLM Learning Module 14: MULTIMODAL / VISION-LANGUAGE MODELS
================================================================================

Techniques for combining vision and language:

1. Vision Transformer (ViT) - Image patch encoding
2. Cross-Attention for multimodal fusion
3. LLaVA-style visual instruction tuning
4. Flamingo-style perceiver resampler

================================================================================
ILLUSTRATION: Vision-Language Architecture
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Vision-Language Model                                 │
    │                                                                          │
    │    Image Input                    Text Input                            │
    │         │                              │                                 │
    │         ▼                              ▼                                 │
    │    ┌─────────────┐               ┌─────────────┐                       │
    │    │   ViT/CLIP  │               │   Tokenizer │                       │
    │    │   Encoder   │               │             │                       │
    │    └─────────────┘               └─────────────┘                       │
    │         │                              │                                 │
    │         ▼                              ▼                                 │
    │    ┌─────────────┐               ┌─────────────┐                       │
    │    │   Image     │               │   Text      │                       │
    │    │   Patches   │               │   Embed     │                       │
    │    │   [N, D]    │               │   [L, D]    │                       │
    │    └─────────────┘               └─────────────┘                       │
    │         │                              │                                 │
    │         └──────────────┬───────────────┘                                 │
    │                        │                                                  │
    │                        ▼                                                  │
    │              ┌─────────────────┐                                         │
    │              │ Cross-Attention │                                         │
    │              │ or Projection   │                                         │
    │              └─────────────────┘                                         │
    │                        │                                                  │
    │                        ▼                                                  │
    │              ┌─────────────────┐                                         │
    │              │  Language Model │                                         │
    │              │  (LLM)          │                                         │
    │              └─────────────────┘                                         │
    │                        │                                                  │
    │                        ▼                                                  │
    │                   Output Text                                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# 1. Vision Transformer (ViT) Components
# =============================================================================

@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.0


class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.

    Args:
        image_size: Size of input image (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        d_model: Embedding dimension

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Patch Embedding Process:                                                  ║
    ║                                                                           ║
    ║  Input Image: 224 × 224 × 3                                               ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  Split into 16 × 16 patches                                               ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  14 × 14 = 196 patches                                                    ║
    ║  Each patch: 16 × 16 × 3 = 768 values                                     ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  Linear projection to d_model                                             ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  Output: 196 × d_model + CLS token                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional projection for patches
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches + 1, d_model)
        """
        batch_size = x.shape[0]

        # Project patches: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add position embeddings
        x = x + self.pos_embed

        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.shape[0]

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(out)


class TransformerBlock(nn.Module):
    """Transformer block for ViT."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.

    Args:
        config: ViTConfig

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  ViT Architecture:                                                         ║
    ║                                                                           ║
    ║  Image (224×224×3)                                                        ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  Patch Embedding (14×14 patches → 196 tokens + CLS)                      ║
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  ┌─────────────────────────────────────────────────────────────┐         │
    ║  │  Transformer Block × 12                                     │         │
    ║  │  ├── LayerNorm                                              │         │
    ║  │  ├── Multi-Head Attention                                   │         │
    ║  │  ├── Add (residual)                                         │         │
    ║  │  ├── LayerNorm                                              │         │
    ║  │  ├── MLP (4× expansion)                                     │         │
    ║  │  └── Add (residual)                                         │         │
    ║  └─────────────────────────────────────────────────────────────┘         │
    ║       │                                                                   ║
    ║       ▼                                                                   ║
    ║  Output: CLS token (D-dimensional) or all patches                        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.d_model,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.num_heads,
                config.mlp_ratio,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_cls: bool = True,
    ) -> torch.Tensor:
        """
        Encode image.

        Args:
            x: Image tensor (batch, channels, height, width)
            return_cls: If True, return only CLS token; else all patches

        Returns:
            Encoded features
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        if return_cls:
            return x[:, 0]  # CLS token
        else:
            return x[:, 1:]  # All patches (no CLS)


# =============================================================================
# 2. Cross-Attention for Multimodal Fusion
# =============================================================================

class CrossAttention(nn.Module):
    """
    Cross-Attention for multimodal fusion.

    Allows text queries to attend to image features.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Cross-Attention vs Self-Attention:                                        ║
    ║                                                                           ║
    ║  Self-Attention:                                                          ║
    ║    Q, K, V all come from same input                                      ║
    ║    Q = K = V = text                                                      ║
    ║                                                                           ║
    ║  Cross-Attention:                                                         ║
    ║    Q comes from text, K and V come from image                            ║
    ║    Q = text, K = V = image_features                                      ║
    ║                                                                           ║
    ║  This allows text to "query" the image for relevant information.         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,  # Q
        image_features: torch.Tensor,  # K, V
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from text to image.

        Args:
            text_features: (batch, text_len, d_model)
            image_features: (batch, image_len, d_model)
            attention_mask: Optional mask for image

        Returns:
            Attended features (batch, text_len, d_model)
        """
        batch_size = text_features.shape[0]

        # Project Q from text, K and V from image
        q = self.w_q(text_features).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(image_features).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(image_features).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(out)


# =============================================================================
# 3. Perceiver Resampler (Flamingo-style)
# =============================================================================

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for variable-length visual features.

    Compresses variable number of image features into fixed number of tokens.
    Used in Flamingo to handle different image resolutions efficiently.

    Args:
        d_model: Model dimension
        num_latents: Number of learned latent queries
        num_layers: Number of perceiver layers
        num_heads: Number of attention heads
        dropout: Dropout probability

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Perceiver Resampler:                                                      ║
    ║                                                                           ║
    ║  Image features (variable N) ──┐                                          ║
    │                                  │                                         │
    │  Learned latents (fixed M)  ────┼──► Cross-Attention ──► Updated latents  │
    │                                  │     (Q=latents, KV=image)              │
    │                                  │                                         │
    │                                  │     Repeat × num_layers                 │
    │                                  │                                         │
    │  Output: M fixed-length tokens   │                                         │
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_latents = num_latents

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossAttention(d_model, num_heads, dropout),
                'norm1': nn.LayerNorm(d_model),
                'self_attn': MultiHeadAttention(d_model, num_heads, dropout),
                'norm2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
                'norm3': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Resample image features to fixed number of tokens.

        Args:
            image_features: (batch, num_image_tokens, d_model)

        Returns:
            resampled: (batch, num_latents, d_model)
        """
        batch_size = image_features.shape[0]

        # Initialize latents
        x = self.latents.expand(batch_size, -1, -1)

        # Apply perceiver layers
        for layer in self.layers:
            # Cross-attention to image features
            x = x + layer['cross_attn'](layer['norm1'](x), image_features)

            # Self-attention among latents
            x = x + layer['self_attn'](layer['norm2'](x), layer['norm2'](x), layer['norm2'](x))

            # MLP
            x = x + layer['mlp'](layer['norm3'](x))

        return x


# =============================================================================
# 4. Vision-Language Model
# =============================================================================

class VisionLanguageModel(nn.Module):
    """
    Simple Vision-Language Model.

    Combines ViT encoder with a language model via cross-attention
    or projection.

    Args:
        vit_config: Vision Transformer config
        d_model_llm: Language model dimension
        num_cross_attn_layers: Number of cross-attention layers
    """

    def __init__(
        self,
        vit_config: ViTConfig,
        d_model_llm: int = 512,
        num_cross_attn_layers: int = 2,
    ):
        super().__init__()

        # Vision encoder
        self.vision_encoder = VisionTransformer(vit_config)

        # Projection to LLM dimension
        self.vision_projection = nn.Linear(vit_config.d_model, d_model_llm)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossAttention(d_model_llm, num_heads=8),
                'norm': nn.LayerNorm(d_model_llm),
            })
            for _ in range(num_cross_attn_layers)
        ])

        # Placeholder for LLM (would be actual language model)
        self.llm_proj = nn.Linear(d_model_llm, d_model_llm)

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (batch, channels, height, width)
            text_embeddings: (batch, seq_len, d_model_llm)

        Returns:
            fused_features: (batch, seq_len, d_model_llm)
        """
        # Encode images
        image_features = self.vision_encoder(images, return_cls=False)  # (B, N, D_vit)

        # Project to LLM dimension
        image_features = self.vision_projection(image_features)  # (B, N, D_llm)

        # Apply cross-attention
        fused = text_embeddings
        for layer in self.cross_attn_layers:
            fused = fused + layer['cross_attn'](layer['norm'](fused), image_features)

        return self.llm_proj(fused)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate multimodal components."""
    print("=" * 80)
    print("MULTIMODAL / VISION-LANGUAGE MODELS DEMONSTRATION")
    print("=" * 80)

    # 1. Patch Embedding
    print("\n" + "-" * 80)
    print("1. PATCH EMBEDDING")
    print("-" * 80)

    batch_size = 2
    image_size = 224
    patch_size = 16
    d_model = 256

    patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)

    # Simulate image input
    images = torch.randn(batch_size, 3, image_size, image_size)
    patches = patch_embed(images)

    num_patches = (image_size // patch_size) ** 2
    print(f"\nInput image shape: {images.shape}")
    print(f"Output patches shape: {patches.shape}")
    print(f"  - {num_patches} patches + 1 CLS token = {num_patches + 1} tokens")

    # 2. Vision Transformer
    print("\n" + "-" * 80)
    print("2. VISION TRANSFORMER (ViT)")
    print("-" * 80)

    vit_config = ViTConfig(
        image_size=224,
        patch_size=16,
        d_model=256,
        num_heads=8,
        num_layers=4,
    )

    vit = VisionTransformer(vit_config)
    vit_params = sum(p.numel() for p in vit.parameters())

    print(f"\nViT configuration:")
    print(f"  - Image size: {vit_config.image_size}")
    print(f"  - Patch size: {vit_config.patch_size}")
    print(f"  - Num patches: {vit.patch_embed.num_patches}")
    print(f"  - d_model: {vit_config.d_model}")
    print(f"  - Num layers: {vit_config.num_layers}")
    print(f"  - Total params: {vit_params:,}")

    # 3. Cross-Attention
    print("\n" + "-" * 80)
    print("3. CROSS-ATTENTION FOR MULTIMODAL FUSION")
    print("-" * 80)

    cross_attn = CrossAttention(d_model=256, num_heads=8)

    text_features = torch.randn(batch_size, 10, 256)  # 10 text tokens
    image_features = torch.randn(batch_size, 196, 256)  # 196 image tokens

    fused = cross_attn(text_features, image_features)

    print(f"\nText features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Fused output shape: {fused.shape}")

    # 4. Perceiver Resampler
    print("\n" + "-" * 80)
    print("4. PERCEIVER RESAMPLER")
    print("-" * 80)

    resampler = PerceiverResampler(
        d_model=256,
        num_latents=64,
        num_layers=4,
        num_heads=8,
    )

    # Variable number of image tokens
    variable_image_features = torch.randn(batch_size, 196, 256)
    resampled = resampler(variable_image_features)

    print(f"\nInput image tokens: {variable_image_features.shape[1]} (variable)")
    print(f"Output latents: {resampled.shape[1]} (fixed)")
    print(f"Compression ratio: {variable_image_features.shape[1] / resampled.shape[1]:.1f}x")

    # 5. Architecture Comparison
    print("\n" + "-" * 80)
    print("5. VISION-LANGUAGE ARCHITECTURES COMPARISON")
    print("-" * 80)
    print("""
    ┌─────────────────┬─────────────────────────┬─────────────────────────────┐
    │ Model           │ Vision Encoder          │ Fusion Method               │
    ├─────────────────┼─────────────────────────┼─────────────────────────────┤
    │ CLIP            │ ViT                     │ Contrastive (separate)      │
    │ Flamingo        │ NFNet + Perceiver       │ Cross-Attention + Gated XATN│
    │ BLIP-2          │ ViT + Q-Former          │ Query Transformer           │
    │ LLaVA           │ CLIP ViT                │ Simple projection           │
    │ GPT-4V          │ Unknown                 │ Unknown                     │
    │ Gemini          │ Unknown                 │ Native multimodal           │
    └─────────────────┴─────────────────────────┴─────────────────────────────┘

    Fusion strategies:
    1. Concatenation: [image_tokens; text_tokens] → LLM
    2. Cross-Attention: Text queries attend to image features
    3. Gated Cross-Attention: Cross-attn with learned gating (Flamingo)
    4. Q-Former: Learnable queries extract image info (BLIP-2)
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. ViT converts images to sequences of patches (tokens)
    2. Cross-attention allows text to "query" image features
    3. Perceiver resampler compresses variable-length visual features
    4. Simple projection (LLaVA) often works surprisingly well
    5. Gated cross-attention (Flamingo) for stable training

    Training tips:
    - Freeze vision encoder initially
    - Use smaller learning rate for cross-attention
    - Pretrain on image-text pairs before instruction tuning

    This completes the LLM Learning Materials!
    All modules: Tokenizer → Embeddings → Attention → RoPE → SwiGLU → MoE
                 → Transformer → Model → Training → Advanced → Efficiency
                 → Fine-tuning → Alignment → Multimodal
    """)


if __name__ == "__main__":
    demo()
