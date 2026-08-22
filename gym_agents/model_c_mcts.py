"""Model C-prior PUCT agent with Fuegi rollouts."""

from __future__ import annotations

from .agents import BaseMonteCarloAgent
from .fuegi import fuegi_trading_strategy, give_dragon_to_opponent_with_more_cards
from .mcts import make_model_c_puct_ismctsearch
from .model_c import ModelCAgent


class ModelCGuidedMctsAgent(BaseMonteCarloAgent):
    """Use Model C probabilities for PUCT exploration, Fuegi for rollouts."""

    def __init__(
        self,
        checkpoint,
        *,
        iterations=100,
        max_time=float("inf"),
        cheat=False,
        device="cpu",
        puct_constant=1.25,
    ):
        self.policy = ModelCAgent(checkpoint, device=device)
        search = make_model_c_puct_ismctsearch(name="ModelCGuidedPuct")
        search.set_action_prior(
            self.policy.action_priors, puct_constant=puct_constant
        )
        super().__init__(
            search_algorithm=search,
            iterations=iterations,
            max_time=max_time,
            cheat=cheat,
        )
        self.trade = fuegi_trading_strategy
        self.give_dragon_away = give_dragon_to_opponent_with_more_cards
        self.puct_constant = puct_constant

    @property
    def info(self):
        return (
            "ModelCGuidedMctsAgent, iterations={}, puct_constant={}, {}"
        ).format(self.iterations, self.puct_constant, self.policy.info)

    def action(self, state):
        # Priors are valid only within one search tree; clearing avoids
        # retaining a large number of old game states across a full match.
        self._search.clear_prior_cache()
        return super().action(state)
