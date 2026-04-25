#!/usr/bin/env python3
"""
================================================================================
LLM Learning - Complete Demonstration Runner
================================================================================

This script runs all the learning modules in sequence, demonstrating the
complete pipeline from tokenization to a trained language model.

Usage:
    python run_all.py [--module MODULE_NUM] [--skip-training]

Options:
    --module MODULE_NUM    Run only a specific module (1-16)
    --skip-training        Skip the training demo (module 9)

================================================================================
"""

import sys
import os
import argparse
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the main banner."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                                                                                 ║
║     █████╗     ███╗   ███╗██╗     ███████╗                                     ║
║    ██╔══██╗    ████╗ ████║██║     ██╔════╝                                     ║
║    ███████║    ██╔████╔██║██║     █████╗                                       ║
║    ██╔══██║    ██║╚██╔╝██║██║     ██╔══╝                                       ║
║    ██║  ██║    ██║ ╚═╝ ██║███████╗███████╗                                     ║
║    ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝╚══════╝                                     ║
║                                                                                 ║
║                    L E A R N I N G   M A T E R I A L S                         ║
║                                                                                 ║
║              Building a Modern LLM from Scratch with PyTorch                    ║
║                                                                                 ║
║                         Target: ~0.1B Parameters                                ║
║                                                                                 ║
║              Techniques: BPE, RoPE, SwiGLU, MoE, Muon, CLIP                     ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


def print_module_header(num: int, title: str):
    """Print a module header."""
    print(f"\n{'=' * 80}")
    print(f"MODULE {num}: {title}")
    print(f"{'=' * 80}\n")


def run_module(module_num: int, skip_training: bool = False):
    """Run a specific module."""
    modules = {
        1: ("fundamentals/tokenizer.py", "TOKENIZER (BPE)"),
        2: ("fundamentals/embeddings.py", "EMBEDDINGS"),
        3: ("fundamentals/attention.py", "ATTENTION MECHANISMS"),
        4: ("position_and_activation/rope.py", "POSITION ENCODINGS (RoPE & ALiBi)"),
        5: ("position_and_activation/swiglu.py", "SwiGLU ACTIVATION"),
        6: ("architecture/transformer_block.py", "TRANSFORMER BLOCK"),
        7: ("architecture/mixture_of_experts.py", "MIXTURE OF EXPERTS"),
        8: ("architecture/complete_model.py", "COMPLETE LLM MODEL"),
        9: ("training/training_loop.py", "TRAINING PIPELINE"),
        10: ("training/optimizers_and_checkpoint.py", "ADVANCED OPTIMIZERS & CHECKPOINTING"),
        11: ("inference/inference_optimization.py", "INFERENCE OPTIMIZATION"),
        12: ("adaptation/parameter_efficient_finetuning.py", "PARAMETER-EFFICIENT FINE-TUNING"),
        13: ("alignment/alignment.py", "MODEL ALIGNMENT"),
        14: ("alignment/grpo.py", "GROUP RELATIVE POLICY OPTIMIZATION"),
        15: ("beyond_text/multimodal.py", "MULTIMODAL / VISION-LANGUAGE MODELS"),
        16: ("beyond_text/mamba_ssm.py", "MAMBA / STATE SPACE MODELS"),
        17: ("inference/mla.py", "MULTI-HEAD LATENT ATTENTION"),
        18: ("architecture/hyper_connections.py", "MANIFOLD-CONSTRAINED HYPER-CONNECTIONS"),
        19: ("architecture/engram.py", "ENGRAM — CONDITIONAL MEMORY"),
    }

    if module_num not in modules:
        print(f"Invalid module number: {module_num}")
        return False

    filename, title = modules[module_num]

    if skip_training and module_num == 9:
        print("Skipping training module...")
        return True

    print_module_header(module_num, title)

    # Import and run the module
    module_path = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(module_path):
        print(f"Module file not found: {module_path}")
        return False

    # Read and execute the module
    with open(module_path, 'r') as f:
        code = f.read()

    # Create a namespace for execution
    namespace = {"__name__": "__main__", "__file__": module_path}

    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Error running module {module_num}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def print_curriculum():
    """Print the curriculum overview."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                            CURRICULUM OVERVIEW                                  ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  FUNDAMENTALS                                                                   ║
║  ─────────────────────                                                          ║
║  • Byte Pair Encoding (BPE) algorithm                                          ║
║  • Training a tokenizer from scratch                                           ║
║  • Token and positional embeddings                                             ║
║  • LayerNorm vs RMSNorm                                                        ║
║  • Scaled dot-product attention                                                 ║
║  • Multi-head and causal attention                                             ║
║                                                                                 ║
║  POSITION & ACTIVATION                                                          ║
║  ─────────────────────                                                          ║
║  • Rotary Position Embeddings (RoPE)                                           ║
║  • ALiBi (Attention with Linear Biases)                                        ║
║  • SwiGLU activation and gated FFN                                             ║
║                                                                                 ║
║  ARCHITECTURE                                                                   ║
║  ─────────────────────                                                          ║
║  • Transformer block with pre-norm and residuals                               ║
║  • Mixture of Experts (MoE) routing                                            ║
║  • Complete LLM assembly (~0.1B params)                                        ║
║                                                                                 ║
║  TRAINING                                                                       ║
║  ─────────────────────                                                          ║
║  • Training loop, LR scheduling, mixed precision                               ║
║  • Muon optimizer (orthogonalized momentum)                                    ║
║  • Gradient checkpointing for memory efficiency                                ║
║                                                                                 ║
║  INFERENCE                                                                      ║
║  ─────────────────────                                                          ║
║  • Flash Attention (memory-efficient)                                          ║
║  • KV Cache for autoregressive generation                                      ║
║  • Sliding Window Attention for long contexts                                  ║
║                                                                                 ║
║  ADAPTATION                                                                     ║
║  ─────────────────────                                                          ║
║  • LoRA (Low-Rank Adaptation)                                                  ║
║  • QLoRA, Prefix Tuning, Adapter layers                                        ║
║                                                                                 ║
║  ALIGNMENT                                                                      ║
║  ─────────────────────                                                          ║
║  • RLHF, DPO, PPO for human preference alignment                               ║
║  • GRPO (Group Relative Policy Optimization)                                   ║
║                                                                                 ║
║  BEYOND TEXT                                                                    ║
║  ─────────────────────                                                          ║
║  • Vision Transformer and multimodal fusion                                    ║
║  • CLIP contrastive learning                                                   ║
║  • Mamba / State Space Models (linear complexity)                              ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


def print_model_architecture():
    """Print the model architecture summary."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                           MODEL ARCHITECTURE                                    ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  Configuration:                                                                 ║
║  ──────────────                                                                 ║
║    vocab_size:   10,000                                                         ║
║    d_model:      256                                                            ║
║    num_heads:    8                                                              ║
║    num_layers:   12                                                             ║
║    d_ff:         683 (~2.67 × d_model)                                         ║
║    max_seq_len:  512                                                            ║
║                                                                                 ║
║  Architecture:                                                                  ║
║  ─────────────                                                                  ║
║    • Token Embeddings (tied with output)                                       ║
║    • 12 × TransformerBlock:                                                    ║
║        - RMSNorm                                                                ║
║        - RoPE Multi-Head Attention (8 heads)                                   ║
║        - Residual Connection                                                    ║
║        - RMSNorm                                                                ║
║        - SwiGLU FFN (d_ff=683)                                                 ║
║        - Residual Connection                                                    ║
║    • Final RMSNorm                                                              ║
║    • Output Linear (tied)                                                       ║
║                                                                                 ║
║  Parameters: ~13M                                                               ║
║                                                                                 ║
║  Techniques Used:                                                               ║
║  ─────────────────                                                              ║
║    ✓ Byte Pair Encoding (BPE)                                                  ║
║    ✓ Rotary Position Embeddings (RoPE)                                         ║
║    ✓ SwiGLU Activation                                                         ║
║    ✓ RMSNorm                                                                   ║
║    ✓ Pre-Norm Architecture                                                     ║
║    ✓ Tied Embeddings                                                           ║
║    ✓ Grouped Query Attention (optional)                                        ║
║    ✓ Mixture of Experts (optional)                                             ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


def print_resources():
    """Print additional learning resources."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                          LEARNING RESOURCES                                     ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  Papers:                                                                        ║
║  ───────                                                                        ║
║  • "Attention Is All You Need" - Original Transformer (2017)                   ║
║  • "Language Models are Few-Shot Learners" - GPT-3 (2020)                      ║
║  • "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)      ║
║  • "GLU Variants Improve Transformer" - SwiGLU (2020)                          ║
║  • "Switch Transformers: Scaling to Trillion Parameter Models" (2021)          ║
║  • "LLaMA: Open and Efficient Foundation Language Models" (2023)               ║
║  • "Learning Transferable Visual Models From Natural Language Supervision"     ║
║    - CLIP (2021)                                                                ║
║  • "Mamba: Linear-Time Sequence Modeling" (2023)                               ║
║  • "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs" (2025)            ║
║                                                                                 ║
║  Code Repositories:                                                             ║
║  ──────────────────                                                             ║
║  • https://github.com/karpathy/nanoGPT - Minimal GPT implementation            ║
║  • https://github.com/facebookresearch/llama - LLaMA                           ║
║  • https://github.com/huggingface/transformers - Hugging Face                  ║
║  • https://github.com/ml-explore/mlx-examples - Apple MLX                      ║
║                                                                                 ║
║  Courses:                                                                       ║
║  ────────                                                                       ║
║  • Andrej Karpathy's "Neural Networks: Zero to Hero"                           ║
║  • Stanford CS224N: NLP with Deep Learning                                      ║
║  • fast.ai "Practical Deep Learning for Coders"                                ║
║                                                                                 ║
║  Blogs:                                                                         ║
║  ───────                                                                        ║
║  • "The Illustrated Transformer" - Jay Alammar                                 ║
║  • "The Annotated Transformer" - Harvard NLP                                   ║
║  • Lilian Weng's Blog - Transformer explanations                               ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Learning Materials")
    parser.add_argument(
        "--module",
        type=int,
        default=None,
        help="Run only a specific module (1-16)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training demo (module 9)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show curriculum overview and exit",
    )
    args = parser.parse_args()

    print_banner()

    if args.list:
        print_curriculum()
        print_model_architecture()
        print_resources()
        return

    start_time = time.time()

    if args.module:
        # Run single module
        run_module(args.module, args.skip_training)
    else:
        # Run all modules
        print("Running all modules...\n")
        print_curriculum()

        input("\nPress Enter to start...")

        for i in range(1, 20):
            if not run_module(i, args.skip_training):
                print(f"\nModule {i} failed. Stopping.")
                break

            if i < 19:
                input("\nPress Enter to continue to next module...")

        print("\n" + "=" * 80)
        print("ALL MODULES COMPLETE!")
        print("=" * 80)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    print_model_architecture()
    print_resources()

    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                                                                                 ║
║                          🎉 Congratulations! 🎉                                  ║
║                                                                                 ║
║               You've completed the LLM Learning Materials!                      ║
║                                                                                 ║
║            You now understand how modern LLMs like GPT, LLaMA,                  ║
║            and others are built from the ground up.                             ║
║                                                                                 ║
║                  Happy building and experimenting!                              ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
