"""PyTorch behavior-cloning baseline for variable Tichu action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from nn_training.imitation_features import FeatureEncoder


@dataclass
class EncodedBatch:
    states: torch.Tensor
    actions: torch.Tensor
    lengths: tuple[int, ...]
    targets: torch.Tensor

    def to(self, device: torch.device) -> "EncodedBatch":
        return EncodedBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            lengths=self.lengths,
            targets=self.targets.to(device),
        )


@dataclass
class PaddedCandidateBatch:
    """Tensor batch with one padded legal-action axis per decision."""

    states: torch.Tensor
    actions: torch.Tensor
    legal_action_mask: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "PaddedCandidateBatch":
        return PaddedCandidateBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            legal_action_mask=self.legal_action_mask.to(device),
            targets=self.targets.to(device),
        )


def encode_batch(records: Sequence[dict], encoder: FeatureEncoder) -> EncodedBatch:
    states = []
    actions = []
    lengths = []
    targets = []
    for record in records:
        state, candidates, target = encoder.encode_record(record)
        states.append(state)
        actions.append(candidates)
        lengths.append(len(candidates))
        targets.append(target)
    return EncodedBatch(
        states=torch.from_numpy(np.stack(states)),
        actions=torch.from_numpy(np.concatenate(actions)),
        lengths=tuple(lengths),
        targets=torch.tensor(targets, dtype=torch.long),
    )


def encode_padded_batch(
    records: Sequence[Mapping], encoder: FeatureEncoder
) -> PaddedCandidateBatch:
    """Encode a DataLoader record batch into dense tensors and a legal mask."""
    if not records:
        raise ValueError("cannot encode an empty record batch")

    encoded = [encoder.encode_record(record) for record in records]
    states = torch.from_numpy(np.stack([item[0] for item in encoded]))
    lengths = [len(item[1]) for item in encoded]
    maximum_actions = max(lengths)
    actions = torch.zeros(
        (len(encoded), maximum_actions, encoder.action_size), dtype=torch.float32
    )
    legal_action_mask = torch.zeros(
        (len(encoded), maximum_actions), dtype=torch.bool
    )
    for row, (_, candidates, _) in enumerate(encoded):
        length = len(candidates)
        actions[row, :length] = torch.from_numpy(candidates)
        legal_action_mask[row, :length] = True
    targets = torch.tensor([item[2] for item in encoded], dtype=torch.long)
    return PaddedCandidateBatch(
        states=states,
        actions=actions,
        legal_action_mask=legal_action_mask,
        targets=targets,
    )


class CandidatePolicy(nn.Module):
    """Score each legal action conditioned on its public observation."""

    def __init__(
        self,
        *,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: EncodedBatch) -> tuple[torch.Tensor, ...]:
        state_embeddings = self.state_encoder(batch.states)
        repeated_states = torch.repeat_interleave(
            state_embeddings,
            torch.tensor(batch.lengths, device=batch.states.device),
            dim=0,
        )
        action_embeddings = self.action_encoder(batch.actions)
        joint = torch.cat(
            (
                repeated_states,
                action_embeddings,
                repeated_states * action_embeddings,
            ),
            dim=1,
        )
        flat_scores = self.scorer(joint).squeeze(1)
        return torch.split(flat_scores, batch.lengths)

    def config(self) -> dict:
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
        }


class BaselineBPolicy(nn.Module):
    """Mask-aware MLP scorer for a state and each legal action candidate."""

    checkpoint_name = "baseline-b-candidate-mlp"

    def __init__(
        self,
        *,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: PaddedCandidateBatch) -> torch.Tensor:
        if batch.states.ndim != 2 or batch.states.shape[1] != self.state_size:
            raise ValueError("states must have shape [batch, state_size]")
        if (
            batch.actions.ndim != 3
            or batch.actions.shape[0] != batch.states.shape[0]
            or batch.actions.shape[2] != self.action_size
        ):
            raise ValueError("actions must have shape [batch, actions, action_size]")
        if batch.legal_action_mask.shape != batch.actions.shape[:2]:
            raise ValueError("legal_action_mask must match the padded action axis")
        if not torch.all(batch.legal_action_mask.any(dim=1)):
            raise ValueError("every decision requires at least one legal action")

        state_embeddings = self.state_encoder(batch.states).unsqueeze(1)
        state_embeddings = state_embeddings.expand(-1, batch.actions.shape[1], -1)
        action_embeddings = self.action_encoder(batch.actions)
        joint = torch.cat(
            (
                state_embeddings,
                action_embeddings,
                state_embeddings * action_embeddings,
                torch.abs(state_embeddings - action_embeddings),
            ),
            dim=-1,
        )
        logits = self.scorer(joint).squeeze(-1)
        return logits.masked_fill(~batch.legal_action_mask, float("-inf"))

    def config(self) -> dict:
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
        }


def batch_loss_and_rankings(
    scores: Sequence[torch.Tensor], targets: torch.Tensor
) -> tuple[torch.Tensor, list[int], list[bool], list[int]]:
    """Calculate a batch loss with only one device synchronization.

    Padding candidate groups lets PyTorch run one cross-entropy and one top-k
    operation per batch.  Converting individual CUDA scalars to Python values
    would otherwise synchronize the GPU once per human decision.
    """
    padded = nn.utils.rnn.pad_sequence(
        scores, batch_first=True, padding_value=float("-inf")
    )
    loss = nn.functional.cross_entropy(padded, targets)
    ranking = torch.topk(padded, k=min(3, padded.shape[1]), dim=1).indices
    predictions = ranking[:, 0].detach().cpu().tolist()
    target_indices = targets.detach().cpu().tolist()
    top3_matches = (
        torch.any(ranking == targets[:, None], dim=1).detach().cpu().tolist()
    )
    return loss, predictions, top3_matches, target_indices
