"""Inference agent for the frozen Model C Set-Attention + Bi-GRU policy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gym_tichu.envs.internals import PlayerAction, TichuState

from nn_training.imitation_features import FeatureSchema
from nn_training.model_c import ModelCEncoder, ModelCPolicy

from .agents import DefaultGymAgent
from .fuegi import fuegi_trading_strategy, give_dragon_to_opponent_with_more_cards
from .training_data import encode_action, encode_observation


logger = logging.getLogger(__name__)


class ModelCAgent(DefaultGymAgent):
    """Rank current legal engine actions with the frozen Model C checkpoint."""

    def __init__(self, checkpoint, *, device="cpu"):
        try:
            import torch
        except ImportError as error:
            raise ImportError("ModelCAgent requires PyTorch training dependencies") from error
        super().__init__(
            trade=fuegi_trading_strategy,
            give_dragon_away=give_dragon_to_opponent_with_more_cards,
        )
        self._torch = torch
        self.checkpoint_path = Path(checkpoint)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        self.device = torch.device(device)
        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if saved.get("model") != "model-c-set-attention-bigru":
            raise ValueError("checkpoint is not a Model C policy")
        schema = FeatureSchema.from_dict(saved["feature_schema"])
        self.encoder = ModelCEncoder(schema)
        self.model = ModelCPolicy(**saved["model_config"]).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.last_scored_actions = ()

    @property
    def info(self):
        return "ModelCAgent, checkpoint={}, device={}".format(self.checkpoint_path, self.device)

    def action(self, state: TichuState) -> PlayerAction:
        actions = sorted(state.possible_actions_list, key=repr)
        if len(actions) == 1:
            self.last_scored_actions = ((actions[0], 0.0),)
            return actions[0]
        priors = self.action_priors(state, actions=actions)
        selected = max(actions, key=lambda action: priors[action])
        logger.debug("Model C action %s prior=%s", selected, priors[selected])
        return selected

    def action_priors(self, state: TichuState, *, actions=None):
        """Return a normalized Model C prior over the supplied legal actions."""
        actions = list(actions or sorted(state.possible_actions_list, key=repr))
        observer = state.player_pos
        encoded_actions = [encode_action(action, observer) for action in actions]
        record = {
            "schema_version": 2,
            "observation": encode_observation(state, observer),
            "legal_actions": encoded_actions,
            "chosen_action": encoded_actions[0],
        }
        batch = self.encoder.encode_batch([record]).to(self.device)
        with self._torch.inference_mode():
            scores = self.model(batch)[0]
        score_values = scores.detach().cpu().tolist()
        self.last_scored_actions = tuple(zip(actions, score_values))
        probabilities = self._torch.softmax(scores, dim=0).detach().cpu().tolist()
        return dict(zip(actions, probabilities))
