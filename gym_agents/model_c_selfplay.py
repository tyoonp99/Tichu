"""Sampling agent and trajectory recorder used by Model C PPO self-play."""

from __future__ import annotations

from dataclasses import replace

import torch

from gym_tichu.envs.internals import PlayerAction, TichuState

from nn_training.model_c import ModelCEncoder
from nn_training.selfplay import ModelCActorCritic, SelfPlayTransition, compute_gae

from .agents import DefaultGymAgent
from .fuegi import fuegi_trading_strategy, give_dragon_to_opponent_with_more_cards
from .training_data import encode_action, encode_observation


class SelfPlayTrajectory:
    """Collect sampled learner decisions and attach a team-relative round reward."""

    def __init__(self):
        self.transitions = []

    def append(self, transition):
        self.transitions.append(transition)

    def reward_round(self, start_index, round_points, *, scale=100.0):
        """Attach a completed round's score only to its final learner action."""
        even_points, odd_points = round_points
        if start_index == len(self.transitions):
            return
        index = len(self.transitions) - 1
        transition = self.transitions[index]
        team_points, opponent_points = (
            (even_points, odd_points)
            if transition.player % 2 == 0
            else (odd_points, even_points)
        )
        self.transitions[index] = replace(
            transition,
            reward=(team_points - opponent_points) / scale,
            done=True,
        )

    def reward_trick(self, start_index, *, winner, trick_points, scale=100.0, weight=1.0):
        """Assign actual captured trick points to the latest learner action.

        The reward is team-relative and deliberately does not invent a heuristic:
        it is positive only when the learner's team receives a scored trick, and
        negative when the opponent receives one.  A zero-point trick is retained
        as a no-op because it carries no score signal under Tichu's rules.
        """
        if start_index == len(self.transitions) or not trick_points:
            return
        index = len(self.transitions) - 1
        transition = self.transitions[index]
        sign = 1.0 if transition.player % 2 == winner % 2 else -1.0
        self.transitions[index] = replace(
            transition,
            reward=transition.reward + sign * weight * trick_points / scale,
        )

    def finish(self, *, gamma=1.0, gae_lambda=0.95):
        """Return round-aware GAE targets for this game's learner decisions."""
        return compute_gae(self.transitions, gamma=gamma, gae_lambda=gae_lambda)


class ModelCSelfPlayAgent(DefaultGymAgent):
    """Sample legal actions from a shared Actor-Critic and record PPO data."""

    def __init__(self, actor_critic: ModelCActorCritic, encoder: ModelCEncoder, trajectory, *, device):
        super().__init__(
            trade=fuegi_trading_strategy,
            give_dragon_away=give_dragon_to_opponent_with_more_cards,
        )
        self.actor_critic = actor_critic
        self.encoder = encoder
        self.trajectory = trajectory
        self.device = torch.device(device)

    def action(self, state: TichuState) -> PlayerAction:
        actions = sorted(state.possible_actions_list, key=repr)
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
            logits, value = self.actor_critic(batch)
            distribution = torch.distributions.Categorical(logits=logits[0])
            selected_index = int(distribution.sample())
            log_prob = float(distribution.log_prob(torch.tensor(selected_index, device=self.device)))
        selected = actions[selected_index]
        record["chosen_action"] = encoded_actions[selected_index]
        self.trajectory.append(
            SelfPlayTransition(
                record=record,
                player=observer,
                old_log_prob=log_prob,
                old_value=float(value[0]),
                reward=0.0,
            )
        )
        return selected
