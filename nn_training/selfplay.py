"""PPO building blocks for self-play fine-tuning of Model C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from nn_training.model_c import ModelCBatch, ModelCEncoder, ModelCPolicy


SELFPLAY_CHECKPOINT_NAME = "model-c-ppo-actor-critic"


class ModelCActorCritic(nn.Module):
    """Model C legal-action actor plus a scalar state-value head."""

    checkpoint_name = SELFPLAY_CHECKPOINT_NAME

    def __init__(self, policy: ModelCPolicy, *, value_hidden_size=None):
        super().__init__()
        self.policy = policy
        width = policy.config()["width"]
        value_hidden_size = value_hidden_size or width
        self.value_hidden_size = value_hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(width, value_hidden_size),
            nn.ReLU(),
            nn.Linear(value_hidden_size, 1),
        )

    @classmethod
    def from_model_c_checkpoint(cls, checkpoint, *, map_location="cpu"):
        saved = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if saved.get("model") != ModelCPolicy.checkpoint_name:
            raise ValueError("checkpoint is not a Model C policy")
        policy = ModelCPolicy(**saved["model_config"])
        policy.load_state_dict(saved["model_state"])
        return cls(policy), saved

    @classmethod
    def from_selfplay_checkpoint(cls, checkpoint, *, map_location="cpu"):
        """Restore an actor-critic checkpoint produced by this module."""
        saved = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if saved.get("model") != SELFPLAY_CHECKPOINT_NAME:
            raise ValueError("checkpoint is not a PPO self-play actor-critic")
        instance = cls(
            ModelCPolicy(**saved["model_config"]["policy_config"]),
            value_hidden_size=saved["model_config"]["value_hidden_size"],
        )
        instance.load_state_dict(saved["model_state"])
        return instance, saved

    def config(self):
        return {
            "policy_config": self.policy.config(),
            "value_hidden_size": self.value_hidden_size,
        }

    def forward(self, batch: ModelCBatch):
        state = self.policy.encode_state(batch)
        return self.policy.score_candidates(batch, state), self.value_head(state).squeeze(-1)


@dataclass(frozen=True)
class SelfPlayTransition:
    """A sampled legal action with the outcome assigned after its round ends."""

    record: Mapping
    player: int
    old_log_prob: float
    old_value: float
    reward: float
    done: bool = False
    advantage: float | None = None
    return_: float | None = None


def compute_gae(transitions: Sequence[SelfPlayTransition], *, gamma=1.0, gae_lambda=0.95):
    """Attach GAE advantages to one learner team's ordered round trajectory."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be in [0, 1]")
    if transitions and not transitions[-1].done:
        raise ValueError("the final self-play transition must terminate a round")

    completed = list(transitions)
    gae = 0.0
    next_value = 0.0
    for index in range(len(completed) - 1, -1, -1):
        transition = completed[index]
        continuation = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * continuation - transition.old_value
        gae = delta + gamma * gae_lambda * continuation * gae
        completed[index] = SelfPlayTransition(
            **{
                **transition.__dict__,
                "advantage": gae,
                "return_": gae + transition.old_value,
            }
        )
        next_value = transition.old_value
    return completed


def ppo_update(
    *,
    actor_critic: ModelCActorCritic,
    reference_policy: ModelCPolicy,
    encoder: ModelCEncoder,
    transitions: Sequence[SelfPlayTransition],
    optimizer,
    device,
    ppo_epochs=2,
    clip_ratio=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    kl_coefficient=0.02,
    minibatch_size=128,
):
    """Run PPO on a fixed self-play batch with a KL anchor to frozen C v2."""
    if not transitions:
        raise ValueError("self-play update requires at least one transition")
    if ppo_epochs < 1:
        raise ValueError("ppo_epochs must be positive")
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")

    # cuDNN GRU requires train mode for its backward pass.  Temporarily disable
    # dropout instead, so PPO ratios use the same deterministic policy as the
    # rollout without giving up recurrent-model gradients on CUDA.
    dropouts = [module for module in actor_critic.modules() if isinstance(module, nn.Dropout)]
    previous_dropout_probabilities = [module.p for module in dropouts]
    actor_critic.train()
    for module in dropouts:
        module.p = 0.0
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    old_values = torch.tensor(
        [transition.old_value for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        [
            transition.return_
            if transition.return_ is not None
            else transition.reward
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.tensor(
        [
            transition.advantage
            if transition.advantage is not None
            else transition.reward - transition.old_value
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-6)

    reference_policy.eval()
    metric_totals = {}
    metric_examples = 0
    try:
        for _ in range(ppo_epochs):
            permutation = torch.randperm(len(transitions), device=device)
            for start in range(0, len(transitions), minibatch_size):
                indexes = permutation[start:start + minibatch_size]
                # Re-encode the records per minibatch so a rare decision with a
                # huge legal-action set cannot pad every other transition.
                batch = encoder.encode_batch(
                    [transitions[index].record for index in indexes.tolist()]
                ).to(device)
                logits, values = actor_critic(batch)
                log_probabilities = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probabilities.gather(1, batch.targets[:, None]).squeeze(1)
                ratios = torch.exp(selected_log_probs - old_log_probs[indexes])
                minibatch_advantages = advantages[indexes]
                unclipped = ratios * minibatch_advantages
                clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * minibatch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[indexes])
                probabilities = log_probabilities.exp()
                entropy = -(probabilities * log_probabilities.masked_fill(~batch.legal_action_mask, 0.0)).sum(dim=1).mean()
                with torch.no_grad():
                    reference_log_probabilities = F.log_softmax(reference_policy(batch), dim=1)
                # Both policies intentionally use -inf for padded candidates.  Masking
                # before multiplication avoids the undefined floating-point product
                # 0 * -inf while preserving the categorical KL on legal actions.
                legal = batch.legal_action_mask
                reference_probabilities = reference_log_probabilities.exp().masked_fill(~legal, 0.0)
                safe_reference_logs = reference_log_probabilities.masked_fill(~legal, 0.0)
                safe_current_logs = log_probabilities.masked_fill(~legal, 0.0)
                kl = (
                    reference_probabilities * (safe_reference_logs - safe_current_logs)
                ).sum(dim=1).mean().clamp_min(0.0)
                loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy + kl_coefficient * kl

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=1.0)
                optimizer.step()
                count = len(indexes)
                for name, value in {
                    "loss": loss, "policy_loss": policy_loss,
                    "value_loss": value_loss, "entropy": entropy, "kl": kl,
                    "mean_ratio": ratios.mean(),
                }.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value.detach()) * count
                metric_examples += count
    finally:
        for module, probability in zip(dropouts, previous_dropout_probabilities):
            module.p = probability
    return {
        **{name: total / metric_examples for name, total in metric_totals.items()},
        "mean_return": float(returns.mean()),
        "mean_advantage": float(advantages.mean()),
        "transitions": len(transitions),
        "minibatch_size": minibatch_size,
    }


def save_selfplay_checkpoint(
    path,
    *,
    actor_critic: ModelCActorCritic,
    feature_schema,
    base_checkpoint,
    update,
    metrics,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model": SELFPLAY_CHECKPOINT_NAME,
            "model_config": actor_critic.config(),
            "model_state": actor_critic.state_dict(),
            "feature_schema": feature_schema.to_dict(),
            "base_checkpoint": str(base_checkpoint),
            "update": update,
            "metrics": metrics,
        },
        path,
    )
