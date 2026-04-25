"""
================================================================================
ENGRAM — CONDITIONAL MEMORY FOR LARGE LANGUAGE MODELS
================================================================================

DeepSeek + PKU (2026). Engram introduces a new axis of sparsity:
conditional MEMORY, complementing MoE's conditional COMPUTATION.

The Core Problem:
    Transformer lacks a native "knowledge lookup" mechanism.
    To recall "Paris is the capital of France", it must CONSUME
    multiple layers of attention + FFN to gradually combine features.
    
    This is wasteful: O(L×d²) computation for facts that could be
    retrieved in O(1) time via a lookup table.

Engram's Solution:
    Insert conditional memory modules into select Transformer layers.
    For each token, extract N-grams → hash → lookup embedding table.
    A context-aware gate filters retrieved memories before injection.

================================================================================
ILLUSTRATION: Engram Architecture
================================================================================

Standard Transformer (recalling a fact):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Token: "Paris"                                                      │
    │     │                                                                │
    │     ▼                                                                │
    │  Layer 1:  Attention ──► "Paris is a city"                          │
    │     │                                                                │
    │     ▼                                                                │
    │  Layer 2:  Attention ──► "Paris is in Europe"                       │
    │     │                                                                │
    │     ▼                                                                │
    │  ...                                                                 │
    │     │                                                                │
    │     ▼                                                                │
    │  Layer 15: Attention ──► "Paris is the capital of France"           │
    │                                                                      │
    │  Cost: 15 layers × full computation                                   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Transformer + Engram:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Token: "Paris"                                                      │
    │     │                                                                │
    │     ├──► Engram Layer 2                                              │
    │     │      │                                                         │
    │     │      ├──► Extract N-grams: ["Paris", "is"]                    │
    │     │      │    Hash ──► Lookup Table ──► Embedding e               │
    │     │      │    Gate: "Is this relevant to current context?"        │
    │     │      │    Inject: h' = h + gate × e                           │
    │     │      │                                                         │
    │     │      └──► "Paris is the capital of France"  (O(1) lookup!)   │
    │     │                                                                │
    │     ▼                                                                │
    │  Layer 3-36: Focus on reasoning, not fact retrieval                  │
    │                                                                      │
    │  Cost: O(1) lookup + 34 layers of reasoning                         │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class TokenCompressor:
    """
    Simplified token compression for Engram.

    Full Engram compresses equivalent tokens (different capitalizations,
    accents, etc.) to canonical forms, reducing vocabulary by ~23%.

    For demonstration, we use a simple hash-based compression.
    """

    @staticmethod
    def compress(token_ids: torch.Tensor) -> torch.Tensor:
        """Simple compression: normalize by small modulo (demo only)."""
        # In practice: NFKC → lowercase → strip accents → collapse whitespace
        # For demo: just return as-is (assume pre-compressed vocabulary)
        return token_ids


class MultiHeadHash(nn.Module):
    """
    Multi-head hashing for N-gram indexing.

    Each N-gram order has K independent hash heads.
    This reduces collision probability and improves retrieval quality.

    Args:
        num_hash_heads: Number of hash heads per N-gram order
        table_size: Size of the embedding lookup table
        max_ngram: Maximum N-gram order (e.g., 3 for up to trigrams)
    """

    def __init__(self, num_hash_heads: int = 8, table_size: int = 10000, max_ngram: int = 3):
        super().__init__()
        self.num_hash_heads = num_hash_heads
        self.table_size = table_size
        self.max_ngram = max_ngram

        # Deterministic hash seeds (learned in full version, fixed for demo)
        # Shape: (max_ngram, num_hash_heads)
        self.register_buffer(
            "hash_seeds",
            torch.randint(1, 100000, (max_ngram, num_hash_heads))
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head hash indices for N-grams.

        Args:
            token_ids: (batch, seq_len) integer token IDs

        Returns:
            indices: (batch, seq_len, num_ngram_orders, num_hash_heads)
                     Hash table indices for each position
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        results = []

        for n in range(1, self.max_ngram + 1):
            # Extract N-grams: for position t, use tokens [t-n+1, ..., t]
            if n == 1:
                # Unigram: just the token itself
                ngram_vals = token_ids.unsqueeze(-1)  # (batch, seq, 1)
            else:
                # N-gram: combine N consecutive tokens
                ngram_list = []
                for i in range(n):
                    shift = n - 1 - i
                    if shift == 0:
                        ngram_list.append(token_ids)
                    else:
                        # Pad with zeros at the beginning
                        padded = F.pad(token_ids, (shift, 0), value=0)
                        ngram_list.append(padded[:, :seq_len])
                # Combine: weighted sum (simple deterministic combination)
                ngram_vals = torch.stack(ngram_list, dim=-1)  # (batch, seq, n)
                # Hash combine: use weighted sum with prime multipliers
                weights = torch.tensor([31**i for i in range(n)], device=device)
                ngram_vals = (ngram_vals * weights).sum(dim=-1, keepdim=True)

            # Multi-head hashing: each head uses different seed
            # hash = ((ngram_val * seed) % large_prime) % table_size
            seeds = self.hash_seeds[n - 1 : n, :]  # (1, num_hash_heads)
            # (batch, seq, 1) * (1, num_hash_heads) → (batch, seq, num_hash_heads)
            indices = ((ngram_vals * seeds) % 99991) % self.table_size
            results.append(indices)

        # Stack: (batch, seq, num_ngram_orders, num_hash_heads)
        return torch.stack(results, dim=-2)  # (batch, seq, max_ngram, num_hash_heads)


class Engram(nn.Module):
    """
    Engram Conditional Memory Module.

    Inserts into select Transformer layers to provide O(1) knowledge lookup.

    Args:
        d_model: Model dimension
        table_size: Size of the N-gram embedding table
        num_hash_heads: Number of hash heads per N-gram order
        max_ngram: Maximum N-gram order (e.g., 2 for bigrams, 3 for trigrams)
        ngram_dim: Dimension of retrieved N-gram embeddings

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Engram Forward Pass:                                                     ║
    ║                                                                           ║
    ║  1. Extract N-grams from token IDs                                      ║
    ║  2. Multi-head hash → lookup table indices                              ║
    ║  3. Retrieve embeddings from table                                      ║
    ║  4. Context-aware gating: gate = σ(⟨h_t, W_K · e_t⟩ / √d)              ║
    ║  5. Inject: h'_t = h_t + gate ⊙ (W_V · e_t)                            ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        d_model: int,
        table_size: int = 10000,
        num_hash_heads: int = 8,
        max_ngram: int = 3,
        ngram_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.table_size = table_size
        self.num_hash_heads = num_hash_heads
        self.max_ngram = max_ngram
        self.ngram_dim = ngram_dim

        # N-gram embedding table (the "memory")
        # In practice this can be 100B+ parameters, stored in CPU/SSD
        self.embedding_table = nn.Embedding(table_size, ngram_dim)

        # Multi-head hasher
        self.hasher = MultiHeadHash(num_hash_heads, table_size, max_ngram)

        # Context-aware gating projections
        self.w_query = nn.Linear(d_model, ngram_dim, bias=False)
        self.w_key = nn.Linear(ngram_dim, ngram_dim, bias=False)
        self.w_value = nn.Linear(ngram_dim, d_model, bias=False)

        # Layer norm for stability
        self.norm = nn.RMSNorm(ngram_dim)

        # Compression ratio tracking
        self._lookup_count = 0
        self._cache_hit_count = 0

    def retrieve(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve N-gram embeddings from the memory table.

        Args:
            token_ids: (batch, seq_len) integer token IDs

        Returns:
            embeddings: (batch, seq_len, num_ngram_orders, num_hash_heads, ngram_dim)
            indices: (batch, seq_len, num_ngram_orders, num_hash_heads)
        """
        batch_size, seq_len = token_ids.shape

        # Compress tokens
        compressed = TokenCompressor.compress(token_ids)

        # Get hash indices
        indices = self.hasher(compressed)  # (batch, seq, num_orders, num_heads)

        # Reshape to separate ngram_orders and hash_heads
        indices = indices.view(batch_size, seq_len, self.max_ngram, self.num_hash_heads)

        # Flatten for embedding lookup
        flat_indices = indices.view(-1)
        flat_embeddings = self.embedding_table(flat_indices)

        # Reshape back
        embeddings = flat_embeddings.view(
            batch_size, seq_len, self.max_ngram, self.num_hash_heads, self.ngram_dim
        )

        return embeddings, indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject conditional memory into hidden states.

        Args:
            hidden_states: (batch, seq_len, d_model) from previous layer
            token_ids: (batch, seq_len) input token IDs

        Returns:
            updated_states: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Retrieve N-gram embeddings
        retrieved, indices = self.retrieve(token_ids)
        # retrieved: (batch, seq, max_ngram, num_hash_heads, ngram_dim)

        # Step 2: Aggregate across hash heads (reduce collision impact)
        # Average over hash heads: (batch, seq, max_ngram, ngram_dim)
        memory = retrieved.mean(dim=3)

        # Step 3: Aggregate across N-gram orders (sum)
        # (batch, seq, ngram_dim)
        memory = memory.sum(dim=2)

        # Step 4: Normalize
        memory = self.norm(memory)

        # Step 5: Context-aware gating
        # Query from current hidden state
        query = self.w_query(hidden_states)  # (batch, seq, ngram_dim)

        # Key from retrieved memory
        key = self.w_key(memory)  # (batch, seq, ngram_dim)

        # Gate: sigmoid of scaled dot product
        gate_input = (query * key).sum(dim=-1) / (self.ngram_dim ** 0.5)
        gate = torch.sigmoid(gate_input).unsqueeze(-1)  # (batch, seq, 1)

        # Step 6: Value projection and injection
        value = self.w_value(memory)  # (batch, seq, d_model)
        updated = hidden_states + gate * value

        return updated

    def get_sparsity_stats(self) -> dict:
        """Return sparsity statistics (for monitoring)."""
        return {
            "table_size": self.table_size,
            "ngram_dim": self.ngram_dim,
            "total_table_params": self.table_size * self.ngram_dim,
            "active_params_per_token": self.num_hash_heads * self.ngram_dim * self.max_ngram,
        }


class TransformerWithEngram(nn.Module):
    """
    Simple Transformer block with optional Engram injection.

    Demonstrates how Engram is inserted at specific layers
    (e.g., layers 2 and 15 in DeepSeek's configuration).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        vocab_size: int,
        engram_layers: Optional[List[int]] = None,
        engram_config: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.engram_layers = engram_layers or []

        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Engram modules at specified layers
        self.engrams = nn.ModuleDict()
        if engram_config is None:
            engram_config = {}
        for layer_idx in self.engram_layers:
            self.engrams[str(layer_idx)] = Engram(d_model, **engram_config)

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward with conditional memory injection."""
        x = self.token_embed(token_ids)

        for i, layer in enumerate(self.layers):
            # Standard transformer layer
            x = layer(x)

            # Engram injection at specific layers
            if str(i) in self.engrams:
                x = self.engrams[str(i)](x, token_ids)

        return self.output(x)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate Engram conditional memory."""
    print("=" * 80)
    print("ENGRAM — CONDITIONAL MEMORY FOR LLMs")
    print("=" * 80)

    d_model = 512
    vocab_size = 1000
    batch_size = 2
    seq_len = 8

    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")

    # Multi-Head Hash
    print("\n" + "-" * 80)
    print("1. MULTI-HEAD HASHING")
    print("-" * 80)

    hasher = MultiHeadHash(num_hash_heads=4, table_size=1000, max_ngram=3)
    indices = hasher(token_ids)

    print(f"\nToken IDs shape: {token_ids.shape}")
    print(f"Hash indices shape: {indices.shape}")
    print(f"  → (batch, seq, ngram_orders={hasher.max_ngram}, hash_heads={hasher.num_hash_heads})")

    # Show first token's hashes
    print(f"\nFirst token (batch=0, pos=0) hash indices:")
    print(f"  Unigram heads: {indices[0, 0, 0, :].tolist()}")
    print(f"  Bigram heads:  {indices[0, 0, 1, :].tolist()}")
    print(f"  Trigram heads: {indices[0, 0, 2, :].tolist()}")
    print("""
    Multiple hash heads reduce collision probability:
    - If head 1 collides, head 2-4 probably don't
    - Averaging across heads gives robust retrieval
    """)

    # Engram module
    print("-" * 80)
    print("2. ENGRAM CONDITIONAL MEMORY")
    print("-" * 80)

    engram = Engram(
        d_model=d_model,
        table_size=5000,
        num_hash_heads=8,
        max_ngram=2,
        ngram_dim=128,
    )

    updated = engram(hidden_states, token_ids)

    print(f"\nInput hidden states:  {hidden_states.shape}")
    print(f"Updated hidden states: {updated.shape}")
    print(f"Change magnitude: {(updated - hidden_states).norm(dim=-1).mean().item():.4f}")

    stats = engram.get_sparsity_stats()
    print(f"""
    Engram Statistics:
    - Table size: {stats['table_size']:,}
    - N-gram dim: {stats['ngram_dim']}
    - Total table params: {stats['total_table_params']:,}
    - Active params per token: {stats['active_params_per_token']}
    """)

    # Gate behavior
    print("-" * 80)
    print("3. CONTEXT-AWARE GATING")
    print("-" * 80)

    # Create two different contexts for the same token
    token_ids_same = torch.full((2, 4), 42)  # Same token everywhere
    context_a = torch.randn(1, 4, d_model)  # Context A
    context_b = torch.randn(1, 4, d_model)  # Context B

    out_a = engram(context_a, token_ids_same[:1])
    out_b = engram(context_b, token_ids_same[:1])

    gate_diff = (out_a - context_a).norm() - (out_b - context_b).norm()
    print(f"""
    Same token (ID=42) with different contexts:
    - Context A injection magnitude: {(out_a - context_a).norm(dim=-1).mean().item():.4f}
    - Context B injection magnitude: {(out_b - context_b).norm(dim=-1).mean().item():.4f}
    - Difference: {gate_diff.item():.4f}

    The gate learned to inject DIFFERENT amounts of memory
    based on the surrounding context!
    """)

    # Full model demonstration
    print("-" * 80)
    print("4. TRANSFORMER WITH ENGRAM INJECTION")
    print("-" * 80)

    model = TransformerWithEngram(
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=6,
        vocab_size=500,
        engram_layers=[2, 4],  # Inject at layers 2 and 4
        engram_config={
            "table_size": 2000,
            "num_hash_heads": 4,
            "max_ngram": 2,
            "ngram_dim": 64,
        },
    )

    input_ids = torch.randint(0, 500, (2, 10))
    output = model(input_ids)

    print(f"\nModel: 6-layer Transformer + Engram at layers 2 and 4")
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    engram_params = sum(p.numel() for p in model.engrams.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Engram parameters: {engram_params:,} ({engram_params/total_params*100:.1f}%)")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Engram provides O(1) knowledge lookup via N-gram hashing
    2. Multi-head hashing reduces collision probability
    3. Context-aware gate filters irrelevant memories
    4. Memory table can be offloaded to CPU/SSD (100B+ params feasible)
    5. Optimal allocation: 20-25% sparse params to memory, 75-80% to compute
    6. Complements MoE: MoE = conditional compute, Engram = conditional memory

    Next: beyond_text/mamba_ssm.py - State Space Models
    """)


if __name__ == "__main__":
    demo()
