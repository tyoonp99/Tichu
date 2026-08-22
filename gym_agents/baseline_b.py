"""Inference agent for the mask-aware Baseline B candidate policy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gym_tichu.envs.internals import PlayerAction, TichuState

from nn_training.imitation_features import FeatureEncoder, FeatureSchema

from .agents import DefaultGymAgent
from .fuegi import fuegi_trading_strategy, give_dragon_to_opponent_with_more_cards
from .training_data import encode_action, encode_observation


logger = logging.getLogger(__name__)


class BaselineBAgent(DefaultGymAgent):
    """Rank the engine's current legal actions with a Baseline B checkpoint."""

    def __init__(self, checkpoint, *, device="cpu"):
        try:
            import torch
            from nn_training.behavior_cloning import BaselineBPolicy, PaddedCandidateBatch
        except ImportError as error:
            raise ImportError("BaselineBAgent requires PyTorch training dependencies") from error

        super().__init__(
            trade=fuegi_trading_strategy,
            give_dragon_away=give_dragon_to_opponent_with_more_cards,
        )
        self._torch = torch
        self._batch_type = PaddedCandidateBatch
        self.checkpoint_path = Path(checkpoint)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        self.device = torch.device(device)

        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if saved.get("format_version") != 2:
            raise ValueError("unsupported Baseline B checkpoint format")
        if saved.get("model") != BaselineBPolicy.checkpoint_name:
            raise ValueError("checkpoint is not a Baseline B policy")
        schema = FeatureSchema.from_dict(saved["feature_schema"])
        self.encoder = FeatureEncoder(schema)
        config = saved["model_config"]
        if config["state_size"] != self.encoder.state_size:
            raise ValueError("checkpoint state feature size does not match its schema")
        if config["action_size"] != self.encoder.action_size:
            raise ValueError("checkpoint action feature size does not match its schema")
        self.model = BaselineBPolicy(**config).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.last_scored_actions = ()

    @property
    def info(self):
        return "BaselineBAgent, checkpoint={}, device={}".format(
            self.checkpoint_path, self.device
        )

    def action(self, state: TichuState) -> PlayerAction:
        actions = sorted(state.possible_actions_list, key=repr)
        if len(actions) == 1:
            self.last_scored_actions = ((actions[0], 0.0),)
            return actions[0]

        observer = state.player_pos
        state_vector = self.encoder.encode_observation(
            encode_observation(state, observer)
        )
        action_vectors = np.stack(
            [
                self.encoder.encode_action(encode_action(action, observer))
                for action in actions
            ]
        )
        batch = self._batch_type(
            states=self._torch.from_numpy(state_vector[None, :]).to(self.device),
            actions=self._torch.from_numpy(action_vectors[None, :, :]).to(self.device),
            legal_action_mask=self._torch.ones(
                (1, len(actions)), dtype=self._torch.bool, device=self.device
            ),
            targets=self._torch.zeros(1, dtype=self._torch.long, device=self.device),
        )
        with self._torch.inference_mode():
            scores = self.model(batch)[0]
        score_values = scores.detach().cpu().tolist()
        self.last_scored_actions = tuple(zip(actions, score_values))
        selected_index = int(np.argmax(score_values))
        selected = actions[selected_index]
        logger.debug("Baseline B action %s score=%s", selected, score_values[selected_index])
        return selected
