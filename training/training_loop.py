"""
================================================================================
TRAINING PIPELINE
================================================================================

This module covers the training pipeline for LLMs:
1. Data preparation and tokenization
2. Training loop with gradient accumulation
3. Learning rate scheduling (cosine with warmup)
4. Gradient clipping
5. Mixed precision training (FP16/BF16)
6. Logging and checkpointing

================================================================================
ILLUSTRATION: Training Pipeline
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          Training Pipeline                               │
    │                                                                          │
    │    Raw Text                                                              │
    │        │                                                                 │
    │        ▼                                                                 │
    │    ┌──────────────────────────────────────┐                             │
    │    │          Tokenization                 │                             │
    │    │   "Hello world" → [15496, 995]        │                             │
    │    └──────────────────────────────────────┘                             │
    │        │                                                                 │
    │        ▼                                                                 │
    │    ┌──────────────────────────────────────┐                             │
    │    │     Create Training Batches           │                             │
    │    │   Sliding window over tokens          │                             │
    │    └──────────────────────────────────────┘                             │
    │        │                                                                 │
    │        ▼                                                                 │
    │    ┌──────────────────────────────────────────────────────────────────┐ │
    │    │                    Training Loop                                  │ │
    │    │                                                                   │ │
    │    │   For each batch:                                                 │ │
    │    │     1. Forward pass → logits, loss                               │ │
    │    │     2. Backward pass → gradients                                 │ │
    │    │     3. Gradient clipping (max_norm)                              │ │
    │    │     4. Optimizer step                                            │ │
    │    │     5. Learning rate schedule update                             │ │
    │    │     6. Log metrics                                               │ │
    │    │                                                                   │ │
    │    └──────────────────────────────────────────────────────────────────┘ │
    │        │                                                                 │
    │        ▼                                                                 │
    │    ┌──────────────────────────────────────┐                             │
    │    │      Save Checkpoint                  │                             │
    │    └──────────────────────────────────────┘                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Learning Rate Schedule
================================================================================

    Learning Rate
         │
    1.0 ┤              ╭──────────────────────────────────────────────────────╮
         │             ╱                                                        ╲
    0.8 ┤            ╱                                                          ╲
         │           ╱                                                            ╲
    0.6 ┤          ╱                                                              ╲
         │         ╱                                                                ╲
    0.4 ┤        ╱                                                                  ╲
         │       ╱                                                                    ╲
    0.2 ┤      ╱                                                                      ╲
         │     ╱                                                                        ╲
    0.0 ┤────╯                                                                          └────────
         └──────────────────────────────────────────────────────────────────────────► Steps
              Warmup        Cosine Decay

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Why Warmup?                                                                │
    │                                                                             │
    │  - Early in training, parameters are random → large gradient updates        │
    │  - Large updates can destabilize training                                   │
    │  - Warmup allows the model to "settle in" before full learning rate         │
    └─────────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    vocab_size: int = 10000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 12
    d_ff: int = 683
    max_seq_len: int = 512

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Precision
    use_amp: bool = True
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100

    # Paths
    output_dir: str = "checkpoints"


# =============================================================================
# Data Preparation
# =============================================================================

class TextDataset(Dataset):
    """
    Dataset for language modeling.

    Creates sliding window samples from a tokenized corpus.

    Args:
        tokens: List of token IDs
        seq_len: Sequence length for each sample
    """

    def __init__(self, tokens: List[int], seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input and target sequences.

        For next-token prediction:
            input:  [t0, t1, t2, t3, ...]
            target: [t1, t2, t3, t4, ...]
        """
        input_seq = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        target_seq = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return input_seq, target_seq


def create_dataloader(
    tokens: List[int],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from tokens."""
    dataset = TextDataset(tokens, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class CosineWarmupScheduler:
    """
    Cosine learning rate schedule with warmup.

    Formula:
        lr = base_lr * min(1, step / warmup_steps) * 0.5 * (1 + cos(π * step / total_steps))

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate (default: 0)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Common Learning Rate Schedules:                                          ║
    ║                                                                           ║
    ║  1. Constant: lr = base_lr                                                ║
    ║  2. Linear decay: lr = base_lr * (1 - step/total)                         ║
    ║  3. Cosine decay: lr = base_lr * 0.5 * (1 + cos(π * step/total))          ║
    ║  4. Cosine with warmup: combine warmup + cosine decay                     ║
    ║  5. Inverse sqrt: lr = base_lr / sqrt(max(step, warmup))                  ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return current value."""
        self.current_step += 1

        # Warmup phase
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Training harness for LLMs.

    Features:
    - Gradient accumulation
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Logging and checkpointing

    Args:
        model: The LLM model to train
        config: Training configuration
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),  # Common for LLMs
        )

        # Scheduler (will set total_steps in train())
        self.scheduler = None

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and config.dtype == "float16" else None
        self.amp_dtype = getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

    def train(
        self,
        train_tokens: List[int],
        eval_tokens: Optional[List[int]] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_tokens: Training token IDs
            eval_tokens: Optional evaluation token IDs

        Returns:
            Dictionary of training metrics
        """
        # Create dataloaders
        train_loader = create_dataloader(
            train_tokens,
            self.config.max_seq_len,
            self.config.batch_size,
            shuffle=True,
        )

        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        # Create scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            self.config.warmup_steps,
            total_steps,
        )

        # Metrics tracking
        metrics = {"train_loss": [], "eval_loss": [], "learning_rate": []}

        print(f"\nStarting training...")
        print(f"  Device: {self.device}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Batch size: {self.config.batch_size} × {self.config.gradient_accumulation_steps} (accumulation)")
        print()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_start = time.time()

            self.model.train()

            for step, (input_ids, labels) in enumerate(train_loader):
                # Move to device
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with mixed precision
                if self.scaler is not None:
                    with autocast():
                        _, loss = self.model(input_ids, labels)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        _, loss = self.model(input_ids, labels)
                        loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # Update learning rate
                    current_lr = self.scheduler.step()
                    self.global_step += 1

                    # Track metrics
                    batch_loss = loss.item() * self.config.gradient_accumulation_steps
                    epoch_loss += batch_loss
                    metrics["train_loss"].append(batch_loss)
                    metrics["learning_rate"].append(current_lr)

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        elapsed = time.time() - epoch_start
                        print(
                            f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {batch_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )

                    # Evaluation
                    if eval_tokens is not None and self.global_step % self.config.eval_interval == 0:
                        eval_loss = self.evaluate(eval_tokens)
                        metrics["eval_loss"].append(eval_loss)
                        print(f"  → Eval loss: {eval_loss:.4f}")

                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_checkpoint("best.pt")

            # End of epoch
            avg_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s\n")

        return metrics

    @torch.no_grad()
    def evaluate(self, eval_tokens: List[int]) -> float:
        """Evaluate the model on a held-out set."""
        eval_loader = create_dataloader(
            eval_tokens,
            self.config.max_seq_len,
            self.config.batch_size,
            shuffle=False,
        )

        self.model.eval()
        total_loss = 0.0

        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            _, loss = self.model(input_ids, labels)
            total_loss += loss.item()

        return total_loss / len(eval_loader)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
        }

        path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        import os
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        print(f"Loaded checkpoint from {path}")


# =============================================================================
# Demo with Synthetic Data
# =============================================================================

def demo():
    """
    Demonstrate the training pipeline with synthetic data.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        TRAINING DEMO                                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("TRAINING PIPELINE DEMONSTRATION")
    print("=" * 80)

    # Import model from previous module
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # For demo, we'll create a minimal model inline
    from dataclasses import dataclass as dc

    @dc
    class MiniConfig:
        vocab_size: int = 1000
        d_model: int = 128
        num_heads: int = 4
        num_layers: int = 4
        d_ff: int = 342
        max_seq_len: int = 128
        dropout: float = 0.1
        num_experts: int = 0
        top_k: int = 2
        tie_embeddings: bool = True

    # Create synthetic data
    print("\n" + "-" * 80)
    print("1. CREATING SYNTHETIC DATA")
    print("-" * 80)

    vocab_size = 1000
    seq_len = 128
    num_samples = 10000

    # Random token sequences (in practice, use real text)
    train_tokens = torch.randint(0, vocab_size, (num_samples * seq_len,)).tolist()
    eval_tokens = torch.randint(0, vocab_size, (1000 * seq_len,)).tolist()

    print(f"\nTraining tokens: {len(train_tokens):,}")
    print(f"Eval tokens: {len(eval_tokens):,}")
    print(f"Vocab size: {vocab_size}")

    # Create model and trainer
    print("\n" + "-" * 80)
    print("2. CREATING MODEL AND TRAINER")
    print("-" * 80)

    # Import the model
    from importlib import util
    model_path = os.path.join(os.path.dirname(__file__), "08_model.py")
    spec = util.spec_from_file_location("model", model_path)
    model_module = util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    config = model_module.LLMConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=342,
        max_seq_len=seq_len,
    )

    model = model_module.LLM(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    training_config = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_epochs=1,
        warmup_steps=10,
        max_grad_norm=1.0,
        log_interval=5,
        use_amp=False,  # Disable for demo compatibility
    )

    trainer = Trainer(model, training_config)

    # Train for a few steps (demo)
    print("\n" + "-" * 80)
    print("3. TRAINING (BRIEF DEMO)")
    print("-" * 80)

    # Only use a subset for quick demo
    demo_train_tokens = train_tokens[:5000]
    demo_eval_tokens = eval_tokens[:1000]

    metrics = trainer.train(demo_train_tokens, demo_eval_tokens)

    # Show training curve data
    print("\n" + "-" * 80)
    print("4. TRAINING METRICS")
    print("-" * 80)

    if metrics["train_loss"]:
        print(f"\nFinal training loss: {metrics['train_loss'][-1]:.4f}")
        print(f"Initial training loss: {metrics['train_loss'][0]:.4f}")
        print(f"Loss reduction: {(metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100:.1f}%")

    # Learning rate schedule visualization
    print("\n" + "-" * 80)
    print("5. LEARNING RATE SCHEDULE VISUALIZATION")
    print("-" * 80)

    if metrics["learning_rate"]:
        lrs = metrics["learning_rate"]
        print(f"\nLearning rate progression (first 20 steps):")
        for i, lr in enumerate(lrs[:20]):
            bar = "█" * int(lr / training_config.learning_rate * 20)
            print(f"  Step {i:3d}: {lr:.2e} {bar}")

    # Best practices summary
    print("\n" + "-" * 80)
    print("6. TRAINING BEST PRACTICES")
    print("-" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Key Training Tips:                                                     │
    │                                                                         │
    │  1. Learning Rate:                                                      │
    │     - Start with 3e-4 for small models                                  │
    │     - Scale down for larger models (3e-5 to 1e-4)                       │
    │     - Always use warmup for transformers                                │
    │                                                                         │
    │  2. Batch Size:                                                         │
    │     - Larger is generally better (more stable gradients)                │
    │     - Use gradient accumulation if memory limited                       │
    │     - Aim for 0.5M - 4M tokens per batch for large models              │
    │                                                                         │
    │  3. Precision:                                                          │
    │     - BF16 preferred if hardware supports it (A100, H100, M1/M2)        │
    │     - FP16 + GradScaler for older GPUs                                 │
    │     - FP32 for stability testing only                                   │
    │                                                                         │
    │  4. Regularization:                                                     │
    │     - Weight decay: 0.01 - 0.1                                         │
    │     - Dropout: 0.1 for small models, 0.0-0.1 for large                 │
    │     - Gradient clipping: 1.0                                           │
    │                                                                         │
    │  5. Checkpointing:                                                      │
    │     - Save every 1000-5000 steps                                        │
    │     - Keep best model by eval loss                                      │
    │     - Save optimizer state for resuming                                 │
    └─────────────────────────────────────────────────────────────────────────┘
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Use cosine learning rate schedule with warmup
    2. Gradient accumulation allows larger effective batch sizes
    3. Mixed precision (BF16/FP16) speeds up training significantly
    4. Gradient clipping prevents exploding gradients
    5. Regular evaluation helps detect overfitting
    6. Save checkpoints regularly to avoid losing progress

    Next: training/optimizers_and_checkpoint.py - Advanced Optimizers
    """)


if __name__ == "__main__":
    import os
    demo()
