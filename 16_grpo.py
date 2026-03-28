"""
================================================================================
LLM Learning Module 16: GRPO (Group Relative Policy Optimization)
================================================================================

GRPO is a reinforcement learning algorithm introduced by DeepSeek that:
- Eliminates the need for a value model (critic)
- Uses group-based reward comparison
- Reduces computational cost significantly
- Powers DeepSeek-R1's reasoning capabilities

Key Difference from PPO:
- PPO needs 3 models: Policy, Reward, Value
- GRPO needs only 2 models: Policy, Reward (no Value model!)

================================================================================
ILLUSTRATION: PPO vs GRPO
================================================================================

    PPO (Proximal Policy Optimization):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Policy Model ──► Generate Response ──► Reward Model ──► Reward   │
    │         │                                      │                    │
    │         │                                      ▼                    │
    │         │                               Value Model ──► Advantage   │
    │         │                                      │                    │
    │         └──────────────────────────────────────┘                    │
    │                            Update Policy                             │
    │                                                                      │
    │    Required Models: Policy + Reward + Value (3 models)              │
    │    Value model is typically same size as policy → EXPENSIVE!        │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    GRPO (Group Relative Policy Optimization):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Policy Model ──► Generate K responses for same prompt            │
    │         │                                                            │
    │         │                                                            │
    │         ▼                                                            │
    │    Reward Model ──► Score each response: [r1, r2, ..., rk]         │
    │         │                                                            │
    │         ▼                                                            │
    │    Compute Advantages using GROUP STATISTICS:                       │
    │         A_i = (r_i - mean(rewards)) / std(rewards)                  │
    │         │                                                            │
    │         └──────────────────────────────────────────────┐            │
    │                                                         │            │
    │                            Update Policy based on group relative     │
    │                                                                      │
    │    Required Models: Policy + Reward (only 2 models!)                │
    │    No value model → MUCH CHEAPER!                                   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: GRPO Algorithm Steps
================================================================================

    Step 1: Sample Multiple Responses
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Prompt: "What is 2 + 3?"                                         │
    │                                                                      │
    │    Generate K=4 responses:                                          │
    │    ┌────────────────────────────────────────────────────────────┐   │
    │    │ Response 1: "The answer is 5."                             │   │
    │    │ Response 2: "2 + 3 = 5"                                     │   │
    │    │ Response 3: "Let me calculate... 2 + 3 equals 5."          │   │
    │    │ Response 4: "I think it's 4."                               │   │
    │    └────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Step 2: Compute Rewards
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Reward Model scores each response:                               │
    │                                                                      │
    │    Response 1: r1 = 0.8  (correct, concise)                        │
    │    Response 2: r2 = 0.9  (correct, direct)                         │
    │    Response 3: r3 = 0.7  (correct, but verbose)                    │
    │    Response 4: r4 = 0.1  (incorrect)                               │
    │                                                                      │
    │    Group Statistics:                                                 │
    │    mean = (0.8 + 0.9 + 0.7 + 0.1) / 4 = 0.625                      │
    │    std = sqrt(variance) ≈ 0.31                                      │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Step 3: Compute Advantages (Group Relative)
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Advantage_i = (r_i - mean) / std                                 │
    │                                                                      │
    │    A1 = (0.8 - 0.625) / 0.31 = +0.56  (above average)              │
    │    A2 = (0.9 - 0.625) / 0.31 = +0.89  (best)                       │
    │    A3 = (0.7 - 0.625) / 0.31 = +0.24  (slightly above)             │
    │    A4 = (0.1 - 0.625) / 0.31 = -1.69  (below average)              │
    │                                                                      │
    │    Key Insight:                                                      │
    │    - Responses better than average get positive advantage           │
    │    - Responses worse than average get negative advantage            │
    │    - We only need RELATIVE quality, not absolute!                   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Step 4: Update Policy
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    For each response i:                                             │
    │                                                                      │
    │    π_θ_new(a_i|s) / π_θ_old(a_i|s) = ratio_i                       │
    │                                                                      │
    │    Loss = -min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i)       │
    │                                                                      │
    │    - If A_i > 0 (good response): increase its probability           │
    │    - If A_i < 0 (bad response): decrease its probability            │
    │    - Clipping prevents too large updates                            │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATION: Why GRPO Works Without Value Model
================================================================================

    Traditional RL (PPO):
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Advantage = Q(s,a) - V(s)                                        │
    │                                                                      │
    │    Where:                                                            │
    │      Q(s,a) = expected reward for action a in state s              │
    │      V(s) = expected reward for state s (averaged over actions)    │
    │                                                                      │
    │    Problem: V(s) requires a value model to estimate!               │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    GRPO's Insight:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Instead of learning V(s), use GROUP MEAN as baseline!            │
    │                                                                      │
    │    Advantage_i = r_i - mean(r_1, r_2, ..., r_k)                     │
    │                                                                      │
    │    This is equivalent to:                                            │
    │      Advantage_i ≈ Q(s,a_i) - V(s)                                  │
    │                                                                      │
    │    Where V(s) is approximated by the average reward of samples      │
    │    from the same state (prompt).                                    │
    │                                                                      │
    │    Key Insight:                                                      │
    │    - Same prompt = same state s                                     │
    │    - Different responses = different actions a                      │
    │    - Average reward ≈ V(s)                                          │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


# =============================================================================
# 1. GRPO Configuration
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    num_samples: int = 4              # K: number of samples per prompt
    clip_range: float = 0.2           # PPO clipping parameter ε
    kl_coef: float = 0.1              # KL divergence coefficient
    entropy_coef: float = 0.01        # Entropy bonus coefficient
    max_length: int = 512             # Maximum response length
    temperature: float = 1.0          # Sampling temperature
    learning_rate: float = 1e-5       # Learning rate
    gamma: float = 1.0                # Discount factor (usually 1.0 for GRPO)


# =============================================================================
# 2. Simple Policy Model (for demonstration)
# =============================================================================

class SimplePolicyModel(nn.Module):
    """
    Simple policy model for demonstration.
    
    In practice, this would be a full language model.
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Simple transformer-like layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4)
            for _ in range(2)
        ])
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for input."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits
    
    def get_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for each token."""
        logits = self.forward(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate response given prompt."""
        generated = prompt_ids.clone()
        
        for _ in range(max_length - prompt_ids.size(1)):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# =============================================================================
# 3. Simple Reward Model (for demonstration)
# =============================================================================

class SimpleRewardModel(nn.Module):
    """
    Simple reward model for demonstration.
    
    In practice, this would be trained on human preferences.
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        
        # Reward head
        self.reward_head = nn.Linear(d_model, 1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get reward for input sequence."""
        x = self.embedding(input_ids)
        x = self.encoder(x)
        # Use last hidden state for reward
        reward = self.reward_head(x[:, -1, :])
        return reward.squeeze(-1)


# =============================================================================
# 4. GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer.
    
    This implements the core GRPO algorithm:
    1. Sample K responses per prompt
    2. Compute rewards for each response
    3. Compute advantages using group statistics
    4. Update policy using PPO-style objective with group-relative advantages
    
    Args:
        policy_model: The language model to train
        reward_model: The reward model
        config: GRPO configuration
        reference_model: Reference model for KL penalty (optional)
    
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  GRPO vs PPO:                                                              ║
    ║                                                                           ║
    ║  PPO:                                                                     ║
    ║    Advantage = Q(s,a) - V(s)  where V(s) from value model               ║
    ║    Needs: Policy + Reward + Value models                                 ║
    ║                                                                           ║
    ║  GRPO:                                                                    ║
    ║    Advantage_i = (r_i - mean(r)) / std(r)                                ║
    ║    Needs: Policy + Reward models only                                    ║
    ║                                                                           ║
    ║  The key insight is that the group mean serves as a baseline            ║
    ║  replacing the learned value function.                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        config: GRPOConfig,
        reference_model: Optional[nn.Module] = None
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config
        self.reference_model = reference_model
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using group statistics.
        
        Args:
            rewards: Rewards for K samples [batch_size, K]
        
        Returns:
            advantages: Group-relative advantages [batch_size, K]
        
        ╔═══════════════════════════════════════════════════════════════════════════╗
        ║  Group Advantage Computation:                                              ║
        ║                                                                           ║
        ║  For each prompt, we have K responses with rewards r_1, ..., r_K         ║
        ║                                                                           ║
        ║  mean = (1/K) * Σ r_i                                                    ║
        ║  std = sqrt((1/K) * Σ (r_i - mean)²)                                     ║
        ║  advantage_i = (r_i - mean) / std                                        ║
        ║                                                                           ║
        ║  This normalizes rewards within each group, so we learn:                 ║
        ║  - Which responses are better than average                               ║
        ║  - Which are worse than average                                          ║
        ║  - Relative to the specific prompt, not globally                         ║
        ╚═══════════════════════════════════════════════════════════════════════════╝
        """
        # Compute mean and std per group (per prompt)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8  # Add small epsilon for stability
        
        # Normalize
        advantages = (rewards - mean) / std
        
        return advantages
    
    def compute_policy_loss(
        self,
        input_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO policy loss.
        
        Args:
            input_ids: Generated responses [batch_size * K, seq_len]
            old_log_probs: Log probs from old policy [batch_size * K, seq_len]
            advantages: Group advantages [batch_size, K]
            attention_mask: Attention mask
        
        Returns:
            loss: Policy loss
            metrics: Dictionary of metrics
        """
        batch_size_times_k, seq_len = input_ids.shape
        K = self.config.num_samples
        batch_size = batch_size_times_k // K
        
        # Get new log probs
        new_log_probs = self.policy_model.get_log_probs(input_ids)
        
        # Get log probs of actual tokens
        # For simplicity, we'll use the log prob of the token at each position
        token_log_probs = new_log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [batch_size * K, seq_len - 1]
        
        old_token_log_probs = old_log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute ratio (policy / old_policy)
        log_ratio = token_log_probs.sum(dim=1) - old_token_log_probs.sum(dim=1)
        ratio = torch.exp(log_ratio)  # [batch_size * K]
        
        # Reshape advantages to match
        advantages_flat = advantages.view(-1)  # [batch_size * K]
        
        # PPO-style clipped objective
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_range,
            1 + self.config.clip_range
        ) * advantages_flat
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Optional: KL penalty
        kl_div = 0.0
        if self.reference_model is not None:
            with torch.no_grad():
                ref_log_probs = self.reference_model.get_log_probs(input_ids)
                kl_div = (new_log_probs - ref_log_probs).sum(dim=-1).mean()
            policy_loss = policy_loss + self.config.kl_coef * kl_div
        
        # Optional: Entropy bonus (encourages exploration)
        entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1).mean()
        policy_loss = policy_loss - self.config.entropy_coef * entropy
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "mean_ratio": ratio.mean().item(),
            "mean_advantage": advantages_flat.mean().item(),
            "entropy": entropy.item(),
        }
        
        if isinstance(kl_div, torch.Tensor):
            metrics["kl_divergence"] = kl_div.item()
        
        return policy_loss, metrics
    
    def train_step(
        self,
        prompts: List[str],
        tokenizer
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.
        
        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer for encoding prompts
        
        Returns:
            metrics: Dictionary of training metrics
        """
        batch_size = len(prompts)
        K = self.config.num_samples
        
        # Tokenize prompts
        prompt_ids = torch.stack([
            torch.tensor(tokenizer.encode(p)[:self.config.max_length])
            for p in prompts
        ])
        
        # Pad prompts to same length
        max_prompt_len = prompt_ids.size(1)
        prompt_ids = F.pad(prompt_ids, (0, self.config.max_length - max_prompt_len))
        
        # Step 1: Generate K responses per prompt
        all_responses = []
        for prompt_id in prompt_ids:
            prompt_id = prompt_id.unsqueeze(0)
            for _ in range(K):
                response = self.policy_model.generate(
                    prompt_id,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature
                )
                all_responses.append(response)
        
        # Stack all responses [batch_size * K, seq_len]
        response_ids = torch.stack([r.squeeze(0) for r in all_responses])
        
        # Step 2: Get old log probs (before update)
        with torch.no_grad():
            old_log_probs = self.policy_model.get_log_probs(response_ids)
        
        # Step 3: Compute rewards
        with torch.no_grad():
            rewards = self.reward_model(response_ids)  # [batch_size * K]
            rewards = rewards.view(batch_size, K)  # [batch_size, K]
        
        # Step 4: Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards)  # [batch_size, K]
        
        # Step 5: Compute loss and update
        loss, metrics = self.compute_policy_loss(
            response_ids,
            old_log_probs,
            advantages
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Add reward statistics
        metrics["mean_reward"] = rewards.mean().item()
        metrics["std_reward"] = rewards.std().item()
        
        return metrics


# =============================================================================
# 5. GRPO with Reasoning (DeepSeek-R1 style)
# =============================================================================

class ReasoningGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer optimized for reasoning tasks.
    
    DeepSeek-R1 uses GRPO to train reasoning capabilities:
    - Generates chain-of-thought responses
    - Rewards correct final answers
    - Optionally rewards correct reasoning steps
    
    Key additions:
    - Process reward models (PRM) for step-by-step feedback
    - Longer generation for reasoning chains
    - Special prompts for reasoning
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        config: GRPOConfig,
        reference_model: Optional[nn.Module] = None,
        process_reward_model: Optional[nn.Module] = None
    ):
        super().__init__(policy_model, reward_model, config, reference_model)
        self.process_reward_model = process_reward_model
    
    def compute_reasoning_rewards(
        self,
        response_ids: torch.Tensor,
        final_answer_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards combining final answer and reasoning process.
        
        Args:
            response_ids: Response token IDs
            final_answer_reward: Reward for final answer correctness
        
        Returns:
            Combined reward
        """
        if self.process_reward_model is None:
            return final_answer_reward
        
        # Get process rewards (simplified - in practice, this would be more complex)
        with torch.no_grad():
            process_rewards = self.process_reward_model(response_ids)
        
        # Combine rewards
        # Typically: 0.7 * final_answer + 0.3 * process
        combined = 0.7 * final_answer_reward + 0.3 * process_rewards
        
        return combined


# =============================================================================
# 6. Demonstration
# =============================================================================

def demo_grpo():
    """Demonstrate GRPO algorithm."""
    print("=" * 70)
    print("GRPO (GROUP RELATIVE POLICY OPTIMIZATION) DEMONSTRATION")
    print("=" * 70)
    
    config = GRPOConfig(
        num_samples=4,
        clip_range=0.2,
        kl_coef=0.1,
        learning_rate=1e-4
    )
    
    print(f"\nGRPO Configuration:")
    print(f"  Number of samples (K): {config.num_samples}")
    print(f"  Clip range (ε): {config.clip_range}")
    print(f"  KL coefficient: {config.kl_coef}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create simple models for demonstration
    vocab_size = 1000
    d_model = 128
    
    policy_model = SimplePolicyModel(vocab_size, d_model)
    reward_model = SimpleRewardModel(vocab_size, d_model)
    
    print(f"\nPolicy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"Reward model parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    # Create trainer
    trainer = GRPOTrainer(policy_model, reward_model, config)
    
    # Demonstrate advantage computation
    print("\n" + "-" * 70)
    print("Group Advantage Computation Example")
    print("-" * 70)
    
    # Simulated rewards for 2 prompts, 4 responses each
    rewards = torch.tensor([
        [0.8, 0.9, 0.7, 0.1],  # Prompt 1: 3 good, 1 bad
        [0.5, 0.4, 0.6, 0.5],  # Prompt 2: all similar
    ])
    
    print(f"\nRewards (2 prompts × 4 responses):")
    print(rewards)
    
    advantages = trainer.compute_group_advantages(rewards)
    print(f"\nGroup-relative advantages:")
    print(advantages)
    
    print("\nInterpretation:")
    print("  Prompt 1: Response 2 (best) gets highest advantage")
    print("            Response 4 (worst) gets most negative advantage")
    print("  Prompt 2: All similar, so advantages are close to 0")
    
    # Demonstrate PPO vs GRPO model count
    print("\n" + "-" * 70)
    print("PPO vs GRPO Model Comparison")
    print("-" * 70)
    
    ppo_params = (
        sum(p.numel() for p in policy_model.parameters()) +  # Policy
        sum(p.numel() for p in reward_model.parameters()) +  # Reward
        sum(p.numel() for p in policy_model.parameters())    # Value (same size as policy)
    )
    
    grpo_params = (
        sum(p.numel() for p in policy_model.parameters()) +  # Policy
        sum(p.numel() for p in reward_model.parameters())    # Reward only
    )
    
    print(f"\nPPO total parameters: {ppo_params:,}")
    print(f"  - Policy model: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  - Reward model: {sum(p.numel() for p in reward_model.parameters()):,}")
    print(f"  - Value model:  {sum(p.numel() for p in policy_model.parameters()):,}")
    
    print(f"\nGRPO total parameters: {grpo_params:,}")
    print(f"  - Policy model: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  - Reward model: {sum(p.numel() for p in reward_model.parameters()):,}")
    print(f"  - Value model:  NONE (uses group mean instead)")
    
    print(f"\nMemory savings: {(1 - grpo_params/ppo_params)*100:.1f}%")
    
    print("\n" + "-" * 70)
    print("Why GRPO Works")
    print("-" * 70)
    print("""
Key Insight:
- In traditional RL, we need V(s) as a baseline to compute advantage
- V(s) represents "expected reward from state s"
- GRPO realizes: if we sample K responses from same prompt (same state),
  the average reward of these K responses approximates V(s)!

Mathematical Equivalence:
- Traditional: Advantage(a) = Q(s,a) - V(s)
- GRPO:       Advantage(a) = Reward(a) - mean(Reward(all_actions))

Benefits:
1. No value model needed → 33% fewer parameters
2. Simpler training pipeline
3. Better for tasks with clear success criteria (math, code)
4. Used successfully in DeepSeek-R1 for reasoning
""")
    
    print("=" * 70)
    print("GRPO is a key innovation enabling efficient reasoning model training!")
    print("=" * 70)


def demo_reasoning_grpo():
    """Demonstrate GRPO for reasoning tasks."""
    print("\n" + "=" * 70)
    print("GRPO FOR REASONING (DeepSeek-R1 Style)")
    print("=" * 70)
    
    print("""
DeepSeek-R1 uses GRPO to train reasoning capabilities:

Training Pipeline:
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  1. Start with base language model                                  │
│                                                                      │
│  2. Generate multiple reasoning chains for each problem:            │
│     Prompt: "Solve: 2 + 3 * 4"                                      │
│     ┌────────────────────────────────────────────────────────────┐  │
│     │ Chain 1: "Let's solve step by step...                      │  │
│     │          First, 3 * 4 = 12                                  │  │
│     │          Then, 2 + 12 = 14                                  │  │
│     │          Answer: 14"                                        │  │
│     │                                                              │  │
│     │ Chain 2: "2 + 3 = 5, 5 * 4 = 20, Answer: 20" (wrong order) │  │
│     └────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  3. Reward based on:                                                │
│     - Final answer correctness (primary)                            │
│     - Reasoning step quality (optional, with PRM)                   │
│                                                                      │
│  4. Use GRPO to update policy:                                      │
│     - Correct chains get positive advantage                         │
│     - Wrong chains get negative advantage                           │
│     - Model learns to produce better reasoning                      │
│                                                                      │
│  5. Iterate until model reliably produces correct reasoning         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Key Results from DeepSeek-R1:
- R1-Zero: Pure RL, no supervised fine-tuning
- R1: SFT + RL (better instruction following)
- Both use GRPO for RL training
- Achieves o1-level reasoning performance

Emergent Behaviors:
- Model learns to verify its own work
- Discovers backtracking and correction
- Develops structured problem-solving approach
- Shows "aha!" moments during training
""")


# =============================================================================
# 7. Simple Tokenizer (for demonstration)
# =============================================================================

class SimpleTokenizer:
    """Simple tokenizer for demonstration."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        # Simple character-level encoding for demo
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([chr(i) if i < 128 else '?' for i in ids])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_grpo()
    demo_reasoning_grpo()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. GRPO eliminates the need for a value model (critic)
   - Traditional PPO: Policy + Reward + Value = 3 models
   - GRPO: Policy + Reward = 2 models
   - 33% fewer parameters!

2. Key Innovation: Group-relative advantages
   - Sample K responses per prompt
   - Use group mean as baseline (instead of learned value function)
   - Advantage_i = (r_i - mean) / std

3. Benefits:
   - Simpler training pipeline
   - Lower computational cost
   - Works well for reasoning tasks
   - Powers DeepSeek-R1

4. Use Cases:
   - Mathematical reasoning
   - Code generation
   - Logical problem solving
   - Any task with clear success criteria

5. Training Tips:
   - Use K=4 to K=8 samples per prompt
   - Clip range ε=0.2 works well
   - Add KL penalty to prevent drift from base model
   - Entropy bonus encourages exploration

6. Comparison:
   - PPO: More stable, but expensive (needs value model)
   - DPO: Simplest, but needs preference data
   - GRPO: Middle ground - no value model, works with rewards
""")
