"""Greedy inference agent for PPO-fine-tuned Model C checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gym_tichu.envs.internals import PlayerAction, TichuState

from nn_training.imitation_features import FeatureSchema
from nn_training.model_c import ModelCEncoder, ModelCPolicy
from nn_training.selfplay import ModelCActorCritic, SELFPLAY_CHECKPOINT_NAME

from .agents import DefaultGymAgent
from .fuegi import fuegi_trading_strategy, give_dragon_to_opponent_with_more_cards
from .training_data import encode_action, encode_observation


class ModelCPPOAgent(DefaultGymAgent):
    """Choose the highest-scoring legal action from a PPO self-play checkpoint."""

    def __init__(self, checkpoint, *, device="cpu"):
        super().__init__(
            trade=fuegi_trading_strategy,
            give_dragon_away=give_dragon_to_opponent_with_more_cards,
        )
        self.checkpoint_path = Path(checkpoint)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        self.device = torch.device(device)
        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if saved.get("model") != SELFPLAY_CHECKPOINT_NAME:
            raise ValueError("checkpoint is not a Model C PPO actor-critic")
        model_config = saved["model_config"]
        policy = ModelCPolicy(**model_config["policy_config"])
        self.model = ModelCActorCritic(
            policy, value_hidden_size=model_config["value_hidden_size"]
        ).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.encoder = ModelCEncoder(FeatureSchema.from_dict(saved["feature_schema"]))
        self.last_scored_actions = ()

    @property
    def info(self):
        return "ModelCPPOAgent, checkpoint={}, device={}".format(
            self.checkpoint_path, self.device
        )

    def action(self, state: TichuState) -> PlayerAction:
        actions = sorted(state.possible_actions_list, key=repr)
        if len(actions) == 1:
            self.last_scored_actions = ((actions[0], 0.0),)
            return actions[0]
        observer = state.player_pos
        encoded_actions = [encode_action(action, observer) for action in actions]
        record = {
            "schema_version": 2,
            "observation": encode_observation(state, observer),
            "legal_actions": encoded_actions,
            "chosen_action": encoded_actions[0],
        }
        batch = self.encoder.encode_batch([record]).to(self.device)
        with torch.inference_mode():
            scores, _ = self.model(batch)
        score_values = scores[0].detach().cpu().tolist()
        self.last_scored_actions = tuple(zip(actions, score_values))
        return actions[int(np.argmax(score_values))]
