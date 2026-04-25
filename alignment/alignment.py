"""
================================================================================
MODEL ALIGNMENT
================================================================================

Techniques to align LLMs with human preferences and values:

1. RLHF (Reinforcement Learning from Human Feedback)
2. DPO (Direct Preference Optimization)
3. PPO (Proximal Policy Optimization)
4. Constitutional AI

================================================================================
ILLUSTRATION: The Alignment Problem
================================================================================

Why do we need alignment?

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Pre-trained LLM behavior:                                         │
    │                                                                      │
    │    User: "How do I make a bomb?"                                     │
    │    LLM:  [Completes the text literally, may provide harmful info]   │
    │                                                                      │
    │    ──────────────────────────────────────────────────────────────    │
    │                                                                      │
    │    Aligned LLM behavior:                                             │
    │                                                                      │
    │    User: "How do I make a bomb?"                                     │
    │    LLM:  "I can't help with that, but I can discuss chemistry..."   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Alignment goals:
    1. Helpful: Assist users with their goals
    2. Harmless: Avoid generating harmful content
    3. Honest: Be truthful and acknowledge uncertainty

================================================================================
ILLUSTRATION: RLHF Pipeline
================================================================================

    Step 1: Supervised Fine-Tuning (SFT)
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Pre-trained Model ──► Fine-tune on instruction data ──► SFT Model │
    │                                                                      │
    │    Instruction: "Translate to French"                               │
    │    Input: "Hello world"                                             │
    │    Output: "Bonjour le monde"                                       │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Step 2: Reward Model Training
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    Prompt: "Write a poem about spring"                              │
    │                                                                      │
    │    Response A: [Poem 1]  ◄── Human prefers this (score: 1)          │
    │    Response B: [Poem 2]  ◄── Human rejects this (score: 0)          │
    │                                                                      │
    │    Train Reward Model to predict: RM(A) > RM(B)                     │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Step 3: RL Fine-tuning with PPO
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
    │    │   Policy    │ ──► │   Reward    │ ──► │   PPO       │          │
    │    │   Model     │     │   Model     │     │   Update    │          │
    │    │  (SFT+RL)   │     │  (frozen)   │     │             │          │
    │    └─────────────┘     └─────────────┘     └─────────────┘          │
    │           │                                       │                  │
    │           └───────────────────────────────────────┘                  │
    │                         Update policy                               │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


# =============================================================================
# 1. Reward Model
# =============================================================================

@dataclass
class RewardModelConfig:
    """Configuration for Reward Model."""
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    vocab_size: int = 10000
    max_seq_len: int = 512
    dropout: float = 0.1


class RewardModel(nn.Module):
    """
    Reward Model for RLHF.

    Takes (prompt, response) pairs and outputs a scalar reward.
    Trained on human preference comparisons.

    Args:
        config: RewardModelConfig

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Reward Model Training:                                                    ║
    ║                                                                           ║
    ║  Given pairs (prompt, chosen, rejected), train to satisfy:               ║
    ║    sigmoid(RM(prompt, chosen) - RM(prompt, rejected)) ≈ 1                ║
    ║                                                                           ║
    ║  Loss = -log(sigmoid(RM_chosen - RM_rejected))                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.d_model,
                config.num_heads,
                config.d_model * 4,
                config.dropout,
                batch_first=True,
            )
            for _ in range(config.num_layers)
        ])

        # Output head (scalar reward)
        self.reward_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward for input sequence.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            rewards: Scalar rewards (batch,)
        """
        # Embed
        x = self.embedding(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)

        # Pool (use last token)
        if attention_mask is not None:
            last_pos = attention_mask.sum(dim=1) - 1
            x = x[torch.arange(x.size(0)), last_pos]
        else:
            x = x[:, -1, :]

        # Compute reward
        reward = self.reward_head(x).squeeze(-1)

        return reward


def compute_preference_loss(
    reward_model: RewardModel,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: Optional[torch.Tensor] = None,
    rejected_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute preference loss for reward model training.

    Args:
        reward_model: The reward model
        chosen_ids: Token IDs for chosen responses
        rejected_ids: Token IDs for rejected responses
        chosen_mask: Attention mask for chosen
        rejected_mask: Attention mask for rejected

    Returns:
        loss: Preference classification loss
        accuracy: Accuracy of preference prediction
    """
    # Get rewards
    chosen_rewards = reward_model(chosen_ids, chosen_mask)
    rejected_rewards = reward_model(rejected_ids, rejected_mask)

    # Compute loss: -log(sigmoid(chosen - rejected))
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()

    # Compute accuracy
    accuracy = (logits > 0).float().mean()

    return loss, accuracy


# =============================================================================
# 2. PPO (Proximal Policy Optimization)
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    gamma: float = 0.99           # Discount factor
    lam: float = 0.95             # GAE lambda
    clip_range: float = 0.2       # PPO clip range
    vf_coef: float = 0.5          # Value function coefficient
    ent_coef: float = 0.01        # Entropy coefficient
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4           # PPO update epochs per batch


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    Proximal Policy Optimization is a policy gradient method that:
    1. Uses a clipped objective to prevent large policy updates
    2. Estimates advantages using GAE (Generalized Advantage Estimation)
    3. Includes value function and entropy bonuses

    Args:
        policy_model: The model being trained
        reward_model: The reward model (frozen)
        ref_model: Reference model for KL penalty (frozen)
        config: PPOConfig

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  PPO Clipped Objective:                                                   ║
    ║                                                                           ║
    ║  ratio = π_new(a|s) / π_old(a|s)                                         ║
    ║                                                                           ║
    ║  L_clip = min(                                                            ║
    ║    ratio × A,                                                             ║
    ║    clip(ratio, 1-ε, 1+ε) × A                                              ║
    ║  )                                                                        ║
    ║                                                                           ║
    ║  Where A is the advantage estimate and ε is the clip range.              ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        ref_model: nn.Module,
        config: PPOConfig,
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.config = config

        self.optimizer = torch.optim.Adam(
            policy_model.parameters(),
            lr=config.learning_rate,
        )

        # Freeze reward and reference models
        for param in reward_model.parameters():
            param.requires_grad = False
        for param in ref_model.parameters():
            param.requires_grad = False

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE computes advantages as:
        A_t = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}

        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: Reward at each step (batch, seq_len)
            values: Value estimates (batch, seq_len)
            dones: Done flags (batch, seq_len)

        Returns:
            advantages: GAE advantages (batch, seq_len)
        """
        gamma = self.config.gamma
        lam = self.config.lam

        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(rewards.shape[1])):
            if t < rewards.shape[1] - 1:
                next_value = values[:, t + 1]
            else:
                next_value = 0

            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
            advantages[:, t] = last_advantage = delta + gamma * lam * (1 - dones[:, t]) * last_advantage

        return advantages

    def compute_kl_penalty(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference.

        KL(π || π_ref) = Σ π(x) × log(π(x) / π_ref(x))

        This prevents the policy from deviating too far from the reference.
        """
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        kl = policy_probs * (torch.log(policy_probs + 1e-10) - torch.log(ref_probs + 1e-10))
        return kl.sum(dim=-1).mean()

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one PPO update step.

        Args:
            input_ids: Token IDs
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            returns: Return targets for value function

        Returns:
            Dictionary of metrics
        """
        # Get current policy outputs
        # (simplified - actual implementation would get logits from model)
        policy_logits = self.policy_model(input_ids)
        ref_logits = self.ref_model(input_ids)

        # Compute new log probs
        new_log_probs = F.log_softmax(policy_logits, dim=-1)

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        clip_range = self.config.clip_range
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # KL penalty
        kl_penalty = self.compute_kl_penalty(policy_logits, ref_logits)

        # Total loss
        total_loss = policy_loss + 0.1 * kl_penalty

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": total_loss.item(),
        }


# =============================================================================
# 3. DPO (Direct Preference Optimization)
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization.

    DPO simplifies RLHF by directly optimizing the policy using preference
    data, without training a separate reward model or using RL.

    Key insight: The optimal policy under the Bradley-Terry preference model
    can be expressed analytically, allowing direct optimization.

    Args:
        policy_model: Model to optimize
        ref_model: Reference model (frozen)
        beta: Temperature for DPO loss (default: 0.1)
        learning_rate: Learning rate

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  DPO Loss Function:                                                       ║
    ║                                                                           ║
    ║  L_DPO = -E[log σ(β × (log(π/π_ref)(y_w|x) - log(π/π_ref)(y_l|x)))]      ║
    ║                                                                           ║
    ║  where:                                                                   ║
    ║    y_w = chosen (winning) response                                       ║
    ║    y_l = rejected (losing) response                                      ║
    ║    β = temperature parameter                                             ║
    ║    σ = sigmoid function                                                  ║
    ║                                                                           ║
    ║  Intuition: Maximize log-likelihood ratio of chosen vs rejected         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta

        self.optimizer = torch.optim.Adam(
            policy_model.parameters(),
            lr=learning_rate,
        )

        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of labels under model.

        Args:
            model: Language model
            input_ids: Full input sequence
            labels: Labels (shifted for next-token prediction)

        Returns:
            Log probabilities (batch,)
        """
        # Simplified - actual implementation would use model's forward
        logits = model(input_ids)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Sum over sequence
        return token_log_probs.sum(dim=1)

    def dpo_loss(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        Args:
            prompt_ids: Prompt token IDs
            chosen_ids: Chosen response IDs
            rejected_ids: Rejected response IDs

        Returns:
            loss: DPO loss
            metrics: Dictionary of metrics
        """
        # Concatenate prompt and response
        chosen_full = torch.cat([prompt_ids, chosen_ids], dim=1)
        rejected_full = torch.cat([prompt_ids, rejected_ids], dim=1)

        # Get log probs from policy
        policy_chosen_logp = self.compute_log_probs(
            self.policy_model, chosen_full, chosen_ids
        )
        policy_rejected_logp = self.compute_log_probs(
            self.policy_model, rejected_full, rejected_ids
        )

        # Get log probs from reference
        with torch.no_grad():
            ref_chosen_logp = self.compute_log_probs(
                self.ref_model, chosen_full, chosen_ids
            )
            ref_rejected_logp = self.compute_log_probs(
                self.ref_model, rejected_full, rejected_ids
            )

        # Compute log ratios
        chosen_log_ratio = policy_chosen_logp - ref_chosen_logp
        rejected_log_ratio = policy_rejected_logp - ref_rejected_logp

        # DPO loss
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            chosen_reward = self.beta * chosen_log_ratio.mean()
            rejected_reward = self.beta * rejected_log_ratio.mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
        }

        return loss, metrics

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one DPO training step."""
        loss, metrics = self.dpo_loss(prompt_ids, chosen_ids, rejected_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics


# =============================================================================
# 4. Constitutional AI (Self-Critique)
# =============================================================================

class ConstitutionalAI:
    """
    Constitutional AI: Self-critique and revision.

    Instead of human feedback, the model critiques its own outputs
    against a set of principles (constitution) and revises them.

    Args:
        model: Language model
        constitution: List of principles

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  Constitutional AI Process:                                                ║
    ║                                                                           ║
    ║  1. Generate initial response                                             ║
    ║     Prompt: "Write about..."                                              ║
    ║     Response: [initial output]                                            ║
    ║                                                                           ║
    ║  2. Self-critique against constitution                                    ║
    ║     Prompt: "Critique this response based on: [principle]"               ║
    ║     Critique: [identifies issues]                                         ║
    ║                                                                           ║
    ║  3. Revise based on critique                                              ║
    ║     Prompt: "Revise the response: [response] Critique: [critique]"       ║
    ║     Revised: [improved output]                                            ║
    ║                                                                           ║
    ║  Example constitution:                                                    ║
    ║  - "Choose the response that is most helpful and harmless"               ║
    ║  - "Choose the response that is most honest"                             ║
    ║  - "Avoid stereotypes and discrimination"                                ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        model: nn.Module,
        constitution: List[str],
    ):
        self.model = model
        self.constitution = constitution

    def generate_response(
        self,
        prompt: str,
        max_length: int = 100,
    ) -> str:
        """Generate initial response."""
        # Simplified - actual implementation uses model.generate()
        return f"[Response to: {prompt}]"

    def critique_response(
        self,
        response: str,
        principle: str,
    ) -> str:
        """Generate critique based on principle."""
        critique_prompt = f"""Please critique the following response based on this principle:
Principle: {principle}
Response: {response}

Critique:"""
        # Simplified
        return f"[Critique based on: {principle}]"

    def revise_response(
        self,
        response: str,
        critique: str,
    ) -> str:
        """Revise response based on critique."""
        revision_prompt = f"""Please revise the following response based on the critique:

Original response: {response}
Critique: {critique}

Revised response:"""
        # Simplified
        return f"[Revised response]"

    def process_with_constitution(
        self,
        prompt: str,
        num_revisions: int = 1,
    ) -> Tuple[str, List[str], List[str]]:
        """
        Process prompt through constitutional AI pipeline.

        Returns:
            final_response: Final revised response
            critiques: List of critiques
            revisions: List of revisions
        """
        # Initial response
        response = self.generate_response(prompt)

        critiques = []
        revisions = []

        # Apply constitution
        for _ in range(num_revisions):
            for principle in self.constitution:
                critique = self.critique_response(response, principle)
                critiques.append(critique)

                revision = self.revise_response(response, critique)
                revisions.append(revision)
                response = revision

        return response, critiques, revisions


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate alignment techniques."""
    print("=" * 80)
    print("MODEL ALIGNMENT DEMONSTRATION")
    print("=" * 80)

    # 1. Reward Model
    print("\n" + "-" * 80)
    print("1. REWARD MODEL")
    print("-" * 80)

    config = RewardModelConfig()
    reward_model = RewardModel(config)

    # Simulate preference data
    batch_size = 4
    seq_len = 32

    chosen_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    chosen_rewards = reward_model(chosen_ids)
    rejected_rewards = reward_model(rejected_ids)

    print(f"\nChosen rewards:  {chosen_rewards.tolist()}")
    print(f"Rejected rewards: {rejected_rewards.tolist()}")

    loss, accuracy = compute_preference_loss(reward_model, chosen_ids, rejected_ids)
    print(f"\nPreference loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.2%}")

    # 2. DPO
    print("\n" + "-" * 80)
    print("2. DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("-" * 80)

    # Create simple models for demo
    policy_model = nn.Linear(256, config.vocab_size)
    ref_model = nn.Linear(256, config.vocab_size)
    ref_model.load_state_dict(policy_model.state_dict())

    dpo_trainer = DPOTrainer(policy_model, ref_model, beta=0.1)

    print("""
    DPO vs RLHF:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RLHF Pipeline:                                                         │
    │    1. Train reward model on preferences                                │
    │    2. Train policy with PPO using reward model                         │
    │    3. Complex, requires RL infrastructure                              │
    │                                                                         │
    │  DPO Pipeline:                                                          │
    │    1. Directly optimize policy on preference data                      │
    │    2. No reward model needed                                           │
    │    3. Simpler, more stable training                                    │
    └─────────────────────────────────────────────────────────────────────────┘

    DPO Benefits:
    - No RL instability issues
    - Simpler implementation
    - Often comparable or better results
    - Faster training
    """)

    # 3. PPO
    print("-" * 80)
    print("3. PPO (PROXIMAL POLICY OPTIMIZATION)")
    print("-" * 80)

    print("""
    PPO Key Concepts:

    1. Clipped Objective:
       - Prevents too large policy updates
       - ratio = π_new / π_old
       - clip(ratio, 1-ε, 1+ε) where ε ≈ 0.2

    2. Generalized Advantage Estimation (GAE):
       - Balances bias and variance in advantage estimation
       - A_t = Σ (γλ)^l × δ_{t+l}

    3. KL Penalty:
       - Keeps policy close to reference
       - Prevents reward hacking

    Hyperparameters:
    - Learning rate: 1e-6 to 1e-5
    - Clip range: 0.2
    - GAE lambda: 0.95
    - Discount (gamma): 0.99
    """)

    # 4. Constitutional AI
    print("-" * 80)
    print("4. CONSTITUTIONAL AI")
    print("-" * 80)

    constitution = [
        "Choose the response that is most helpful and harmless",
        "Choose the response that avoids stereotypes",
        "Choose the response that is most honest",
    ]

    print(f"\nExample Constitution:")
    for i, principle in enumerate(constitution, 1):
        print(f"  {i}. {principle}")

    print("""
    Constitutional AI Advantages:
    - No human labeling needed
    - Scalable to many principles
    - Model self-improves
    - Transparent principles
    """)

    # Summary
    print("\n" + "-" * 80)
    print("ALIGNMENT TECHNIQUES COMPARISON")
    print("-" * 80)
    print("""
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │ Method          │ Complexity      │ Data Needed     │ Quality         │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ SFT             │ Low             │ Instructions    │ Baseline        │
    │ RLHF (PPO)      │ High            │ Preferences     │ ★★★★☆           │
    │ DPO             │ Medium          │ Preferences     │ ★★★★☆           │
    │ Constitutional  │ Medium          │ Principles      │ ★★★☆☆           │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘

    Recommended Pipeline:
    1. Start with SFT on instruction data
    2. Use DPO for preference alignment (simpler)
    3. Consider RLHF if DPO insufficient
    4. Add Constitutional AI for safety
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Alignment makes models helpful, harmless, and honest
    2. RLHF: Reward model + PPO, most established method
    3. DPO: Simpler alternative, directly optimize on preferences
    4. Constitutional AI: Self-critique without human feedback
    5. Start with DPO, add complexity only if needed

    Next: alignment/grpo.py - Group Relative Policy Optimization
    """)


if __name__ == "__main__":
    demo()
