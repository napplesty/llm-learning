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
    --module MODULE_NUM    Run only a specific module (1-10)
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
        1: ("01_tokenizer.py", "TOKENIZER (BPE)"),
        2: ("02_embeddings.py", "EMBEDDINGS"),
        3: ("03_attention.py", "ATTENTION"),
        4: ("04_rope.py", "ROTARY POSITION EMBEDDINGS"),
        5: ("05_swiglu.py", "SwiGLU ACTIVATION"),
        6: ("06_moe.py", "MIXTURE OF EXPERTS"),
        7: ("07_transformer.py", "TRANSFORMER BLOCK"),
        8: ("08_model.py", "COMPLETE LLM MODEL"),
        9: ("09_training.py", "TRAINING PIPELINE"),
        10: ("10_advanced.py", "ADVANCED TOPICS"),
        11: ("11_efficiency.py", "EFFICIENCY TECHNIQUES"),
        12: ("12_finetuning.py", "FINE-TUNING (LoRA/DPO)"),
        13: ("13_alignment.py", "MODEL ALIGNMENT"),
        14: ("14_multimodal.py", "MULTIMODAL MODELS"),
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
║  Module 1: TOKENIZER                                                           ║
║  ─────────────────────                                                          ║
║  • Byte Pair Encoding (BPE) algorithm                                          ║
║  • Training a tokenizer from scratch                                           ║
║  • Encoding and decoding text                                                   ║
║                                                                                 ║
║  Module 2: EMBEDDINGS                                                          ║
║  ─────────────────────                                                          ║
║  • Token embeddings                                                             ║
║  • Positional embeddings (sinusoidal vs learned)                               ║
║  • RMSNorm vs LayerNorm                                                        ║
║                                                                                 ║
║  Module 3: ATTENTION                                                           ║
║  ─────────────────────                                                          ║
║  • Scaled dot-product attention                                                 ║
║  • Multi-head attention                                                         ║
║  • Causal (autoregressive) attention                                            ║
║  • Grouped Query Attention (GQA)                                               ║
║                                                                                 ║
║  Module 4: ROTARY POSITION EMBEDDINGS (RoPE)                                   ║
║  ─────────────────────────────────────────                                     ║
║  • Rotation-based position encoding                                            ║
║  • Relative position properties                                                ║
║  • Implementation details                                                       ║
║                                                                                 ║
║  Module 5: SwiGLU ACTIVATION                                                   ║
║  ─────────────────────────                                                      ║
║  • Gated Linear Units (GLU)                                                    ║
║  • Swish/SiLU activation                                                        ║
║  • Comparison with standard FFN                                                ║
║                                                                                 ║
║  Module 6: MIXTURE OF EXPERTS (MoE)                                            ║
║  ────────────────────────────────                                              ║
║  • Expert networks                                                              ║
║  • Top-k routing                                                                ║
║  • Load balancing loss                                                          ║
║  • Switch Transformer                                                           ║
║                                                                                 ║
║  Module 7: TRANSFORMER BLOCK                                                   ║
║  ─────────────────────────                                                      ║
║  • Combining all components                                                     ║
║  • Pre-norm architecture                                                        ║
║  • Residual connections                                                         ║
║                                                                                 ║
║  Module 8: COMPLETE LLM MODEL                                                  ║
║  ─────────────────────────                                                      ║
║  • Full model assembly                                                          ║
║  • Parameter counting                                                           ║
║  • Text generation                                                              ║
║                                                                                 ║
║  Module 9: TRAINING PIPELINE                                                   ║
║  ─────────────────────────                                                      ║
║  • Data preparation                                                             ║
║  • Learning rate scheduling                                                     ║
║  • Gradient accumulation                                                        ║
║  • Mixed precision training                                                     ║
║                                                                                 ║
║  Module 10: ADVANCED TOPICS                                                    ║
║  ─────────────────────────                                                      ║
║  • Muon optimizer (momentum + orthogonalization)                               ║
║  • CLIP contrastive learning                                                   ║
║  • Modern techniques overview                                                  ║
║                                                                                 ║
║  Module 11: EFFICIENCY TECHNIQUES                                              ║
║  ───────────────────────────────                                               ║
║  • Flash Attention (memory-efficient)                                          ║
║  • KV Cache for generation                                                     ║
║  • Sliding Window Attention                                                    ║
║  • ALiBi positional encoding                                                   ║
║  • Gradient checkpointing                                                      ║
║                                                                                 ║
║  Module 12: FINE-TUNING (PEFT)                                                 ║
║  ──────────────────────────────                                                ║
║  • LoRA (Low-Rank Adaptation)                                                  ║
║  • QLoRA (Quantized LoRA)                                                      ║
║  • Prefix Tuning                                                               ║
║  • Adapter layers                                                              ║
║                                                                                 ║
║  Module 13: MODEL ALIGNMENT                                                    ║
║  ───────────────────────────                                                   ║
║  • RLHF (Reinforcement Learning from Human Feedback)                           ║
║  • PPO (Proximal Policy Optimization)                                          ║
║  • DPO (Direct Preference Optimization)                                        ║
║  • Constitutional AI                                                           ║
║                                                                                 ║
║  Module 14: MULTIMODAL MODELS                                                  ║
║  ────────────────────────────                                                  ║
║  • Vision Transformer (ViT)                                                    ║
║  • Patch embeddings                                                            ║
║  • Cross-attention for multimodal fusion                                       ║
║  • Perceiver resampler (Flamingo-style)                                        ║
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
        help="Run only a specific module (1-10)",
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

        for i in range(1, 15):
            if not run_module(i, args.skip_training):
                print(f"\nModule {i} failed. Stopping.")
                break

            if i < 14:
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
