"""Token batches and Set-Attention + Bi-GRU candidate policy for Model C."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from nn_training.decision_dataset import JsonlDecisionDataset
from nn_training.imitation_features import FeatureSchema


CARD_SLOTS = 14


def _card_parts(card):
    if "_" not in card:
        return card, "SPECIAL"
    return tuple(card.split("_", 1))


class ModelCEncoder:
    """Convert a JSONL decision into padded card-set, trick-sequence and action tokens."""

    def __init__(self, schema: FeatureSchema, *, target_weights=None):
        self.schema = schema
        self._card = {card: index + 1 for index, card in enumerate(schema.cards)}
        self._action_type = {
            value: index + 1 for index, value in enumerate(schema.action_types)
        }
        self._combination = {
            value: index + 1 for index, value in enumerate(schema.combinations)
        }
        self._wish = {value: index + 1 for index, value in enumerate(schema.wishes)}
        ranks = sorted({_card_parts(card)[0] for card in schema.cards})
        suits = sorted({_card_parts(card)[1] for card in schema.cards})
        rank_index = {value: index + 1 for index, value in enumerate(ranks)}
        suit_index = {value: index + 1 for index, value in enumerate(suits)}
        self.card_rank_ids = tuple(
            [0] + [rank_index[_card_parts(card)[0]] for card in schema.cards]
        )
        self.card_suit_ids = tuple(
            [0] + [suit_index[_card_parts(card)[1]] for card in schema.cards]
        )
        self.rank_vocab_size = len(ranks) + 1
        self.suit_vocab_size = len(suits) + 1
        self.public_size = 4 + 4 + 4 + 4 + 4 + 4 + len(schema.wishes) + 1 + 2
        self.target_weights = dict(target_weights or {})

    def model_token_config(self):
        return {
            "card_vocab_size": len(self.schema.cards) + 1,
            "rank_vocab_size": self.rank_vocab_size,
            "suit_vocab_size": self.suit_vocab_size,
            "action_type_vocab_size": len(self.schema.action_types) + 1,
            "combination_vocab_size": len(self.schema.combinations) + 1,
            "public_size": self.public_size,
            "card_rank_ids": self.card_rank_ids,
            "card_suit_ids": self.card_suit_ids,
        }

    def _cards(self, cards):
        ids = np.zeros(CARD_SLOTS, dtype=np.int64)
        count = min(len(cards), CARD_SLOTS)
        ids[:count] = [self._card.get(card, 0) for card in cards[:count]]
        return ids, ids != 0

    def _action(self, action):
        card_ids, card_mask = self._cards(action.get("cards", ()))
        numeric = np.array(
            [
                float(action.get("height") or 0.0) / 15.0,
                float(bool(action.get("announce", False))),
                float(bool(action.get("grand", False))),
                float(action.get("trick_points", 0)) / 100.0,
                float(self._wish.get(action.get("wish"), 0))
                / max(1, len(self._wish)),
            ],
            dtype=np.float32,
        )
        player = action.get("player")
        return (
            self._action_type.get(action.get("type"), 0),
            self._combination.get(action.get("combination", "NONE"), 0),
            player + 1 if player in range(4) else 0,
            card_ids,
            card_mask,
            numeric,
        )

    def _public(self, observation):
        vector = np.zeros(self.public_size, dtype=np.float32)
        offset = 0
        for values, scale in (
            (observation.get("hand_sizes", ()), 14.0),
            (observation.get("won_trick_counts", ()), 14.0),
            (observation.get("won_trick_points", ()), 200.0),
        ):
            for index, value in enumerate(values[:4]):
                vector[offset + index] = float(value) / scale
            offset += 4
        for key in ("ranking", "announced_tichu", "announced_grand_tichu"):
            for position in observation.get(key, ()):
                if position in range(4):
                    vector[offset + position] = 1.0
            offset += 4
        wish = self._wish.get(observation.get("wish"), 0)
        if wish:
            vector[offset + wish - 1] = 1.0
        offset += len(self._wish)
        vector[offset] = float(wish == 0)
        offset += 1
        vector[offset] = float(observation.get("decision_context") == "bomb_response")
        vector[offset + 1] = float(bool(observation.get("bomb_trick_finish", False)))
        return vector

    def encode_batch(self, records: Sequence[Mapping]):
        if not records:
            raise ValueError("cannot encode an empty Model C batch")
        encoded = []
        max_candidates = max(len(record["legal_actions"]) for record in records)
        for record in records:
            observation = record["observation"]
            hand_ids, hand_mask = self._cards(observation.get("hand", ()))
            trick = list(observation.get("trick", ()))[-self.schema.max_trick_actions :]
            try:
                target = record["legal_actions"].index(record["chosen_action"])
            except ValueError as error:
                raise ValueError("chosen action is absent from legal actions") from error
            chosen = record["chosen_action"]
            target_weight = float(self.target_weights.get(chosen.get("combination"), 1.0))
            encoded.append(
                (hand_ids, hand_mask, self._public(observation), trick, record["legal_actions"], target, target_weight)
            )
        batch_size = len(encoded)
        trick_length = self.schema.max_trick_actions
        action_type = np.zeros((batch_size, trick_length), dtype=np.int64)
        combination = np.zeros_like(action_type)
        player = np.zeros_like(action_type)
        trick_cards = np.zeros((batch_size, trick_length, CARD_SLOTS), dtype=np.int64)
        trick_card_mask = np.zeros_like(trick_cards, dtype=bool)
        trick_numeric = np.zeros((batch_size, trick_length, 5), dtype=np.float32)
        trick_mask = np.zeros((batch_size, trick_length), dtype=bool)
        candidate_type = np.zeros((batch_size, max_candidates), dtype=np.int64)
        candidate_combination = np.zeros_like(candidate_type)
        candidate_player = np.zeros_like(candidate_type)
        candidate_cards = np.zeros((batch_size, max_candidates, CARD_SLOTS), dtype=np.int64)
        candidate_card_mask = np.zeros_like(candidate_cards, dtype=bool)
        candidate_numeric = np.zeros((batch_size, max_candidates, 5), dtype=np.float32)
        legal_mask = np.zeros((batch_size, max_candidates), dtype=bool)
        for row, (_, _, _, trick, legal_actions, _, _) in enumerate(encoded):
            for column, action in enumerate(trick):
                values = self._action(action)
                action_type[row, column], combination[row, column], player[row, column] = values[:3]
                trick_cards[row, column], trick_card_mask[row, column], trick_numeric[row, column] = values[3:]
                trick_mask[row, column] = True
            for column, action in enumerate(legal_actions):
                values = self._action(action)
                candidate_type[row, column], candidate_combination[row, column], candidate_player[row, column] = values[:3]
                candidate_cards[row, column], candidate_card_mask[row, column], candidate_numeric[row, column] = values[3:]
                legal_mask[row, column] = True
        return ModelCBatch(
            hand_cards=torch.as_tensor(np.stack([item[0] for item in encoded])),
            hand_mask=torch.as_tensor(np.stack([item[1] for item in encoded])),
            public=torch.as_tensor(np.stack([item[2] for item in encoded])),
            trick_type=torch.as_tensor(action_type), trick_combination=torch.as_tensor(combination),
            trick_player=torch.as_tensor(player), trick_cards=torch.as_tensor(trick_cards),
            trick_card_mask=torch.as_tensor(trick_card_mask), trick_numeric=torch.as_tensor(trick_numeric),
            trick_mask=torch.as_tensor(trick_mask), candidate_type=torch.as_tensor(candidate_type),
            candidate_combination=torch.as_tensor(candidate_combination), candidate_player=torch.as_tensor(candidate_player),
            candidate_cards=torch.as_tensor(candidate_cards), candidate_card_mask=torch.as_tensor(candidate_card_mask),
            candidate_numeric=torch.as_tensor(candidate_numeric), legal_action_mask=torch.as_tensor(legal_mask),
            targets=torch.tensor([item[5] for item in encoded], dtype=torch.long),
            target_weights=torch.tensor([item[6] for item in encoded], dtype=torch.float32),
        )


@dataclass
class ModelCBatch:
    hand_cards: torch.Tensor; hand_mask: torch.Tensor; public: torch.Tensor
    trick_type: torch.Tensor; trick_combination: torch.Tensor; trick_player: torch.Tensor
    trick_cards: torch.Tensor; trick_card_mask: torch.Tensor; trick_numeric: torch.Tensor; trick_mask: torch.Tensor
    candidate_type: torch.Tensor; candidate_combination: torch.Tensor; candidate_player: torch.Tensor
    candidate_cards: torch.Tensor; candidate_card_mask: torch.Tensor; candidate_numeric: torch.Tensor
    legal_action_mask: torch.Tensor; targets: torch.Tensor; target_weights: torch.Tensor

    def to(self, device):
        return ModelCBatch(**{name: value.to(device) for name, value in self.__dict__.items()})


def make_model_c_dataloader(path, *, encoder, batch_size, max_records=None, shuffle_buffer=1, seed=0, num_workers=0):
    dataset = JsonlDecisionDataset(path, max_records=max_records, shuffle_buffer=shuffle_buffer, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=encoder.encode_batch)


class _SetBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.norm2 = nn.LayerNorm(width)

    def forward(self, values, padding_mask):
        attended, _ = self.attention(values, values, values, key_padding_mask=padding_mask, need_weights=False)
        values = self.norm1(values + attended)
        return self.norm2(values + self.feed_forward(values))


class ModelCPolicy(nn.Module):
    """Set-Attention hand encoder, Bi-GRU trick encoder and masked action scorer."""

    checkpoint_name = "model-c-set-attention-bigru"

    def __init__(self, *, card_vocab_size, rank_vocab_size, suit_vocab_size, action_type_vocab_size, combination_vocab_size, public_size, card_rank_ids, card_suit_ids, width=128, heads=4, dropout=0.1):
        super().__init__()
        self._config = dict(card_vocab_size=card_vocab_size, rank_vocab_size=rank_vocab_size, suit_vocab_size=suit_vocab_size, action_type_vocab_size=action_type_vocab_size, combination_vocab_size=combination_vocab_size, public_size=public_size, card_rank_ids=list(card_rank_ids), card_suit_ids=list(card_suit_ids), width=width, heads=heads, dropout=dropout)
        self.card_embedding = nn.Embedding(card_vocab_size, width, padding_idx=0)
        self.rank_embedding = nn.Embedding(rank_vocab_size, width, padding_idx=0)
        self.suit_embedding = nn.Embedding(suit_vocab_size, width, padding_idx=0)
        self.register_buffer("card_rank_ids", torch.tensor(card_rank_ids, dtype=torch.long))
        self.register_buffer("card_suit_ids", torch.tensor(card_suit_ids, dtype=torch.long))
        self.action_type_embedding = nn.Embedding(action_type_vocab_size, 24, padding_idx=0)
        self.combination_embedding = nn.Embedding(combination_vocab_size, 24, padding_idx=0)
        self.player_embedding = nn.Embedding(5, 12, padding_idx=0)
        self.action_project = nn.Sequential(nn.Linear(width + 24 + 24 + 12 + 5, width), nn.ReLU(), nn.Linear(width, width))
        self.hand_blocks = nn.ModuleList([_SetBlock(width, heads, dropout) for _ in range(2)])
        self.hand_query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.hand_query, std=0.02)
        self.hand_pool = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.position_embedding = nn.Embedding(13, width)
        self.trick_gru = nn.GRU(width, width, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.public_encoder = nn.Sequential(nn.Linear(public_size, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(width * 4, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(width * 4, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, 1))

    def config(self): return self._config.copy()

    def _card_tokens(self, card_ids):
        return self.card_embedding(card_ids) + self.rank_embedding(self.card_rank_ids[card_ids]) + self.suit_embedding(self.card_suit_ids[card_ids])

    def _actions(self, action_type, combination, player, cards, card_mask, numeric):
        card_tokens = self._card_tokens(cards)
        denominator = card_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        card_summary = (card_tokens * card_mask.unsqueeze(-1)).sum(dim=-2) / denominator
        return self.action_project(torch.cat((card_summary, self.action_type_embedding(action_type), self.combination_embedding(combination), self.player_embedding(player), numeric), dim=-1))

    def forward(self, batch: ModelCBatch):
        if not torch.all(batch.legal_action_mask.any(dim=1)):
            raise ValueError("every decision requires one legal action")
        hand = self._card_tokens(batch.hand_cards)
        hand_padding = ~batch.hand_mask
        for block in self.hand_blocks:
            hand = block(hand, hand_padding)
        query = self.hand_query.expand(hand.shape[0], -1, -1)
        hand_summary, _ = self.hand_pool(query, hand, hand, key_padding_mask=hand_padding, need_weights=False)
        trick = self._actions(batch.trick_type, batch.trick_combination, batch.trick_player, batch.trick_cards, batch.trick_card_mask, batch.trick_numeric)
        positions = torch.arange(trick.shape[1], device=trick.device)
        trick = trick + self.position_embedding(positions)[None, :, :]
        lengths = batch.trick_mask.sum(dim=1)
        packed = pack_padded_sequence(trick, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.trick_gru(packed)
        trick_summary = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        trick_summary = trick_summary * lengths.gt(0).unsqueeze(-1)
        state = self.state_encoder(torch.cat((hand_summary.squeeze(1), trick_summary, self.public_encoder(batch.public)), dim=-1))
        candidates = self._actions(batch.candidate_type, batch.candidate_combination, batch.candidate_player, batch.candidate_cards, batch.candidate_card_mask, batch.candidate_numeric)
        state = state.unsqueeze(1).expand_as(candidates)
        joint = torch.cat((state, candidates, state * candidates, torch.abs(state - candidates)), dim=-1)
        return self.scorer(joint).squeeze(-1).masked_fill(~batch.legal_action_mask, float("-inf"))
