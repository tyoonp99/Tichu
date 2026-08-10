"""Explainable strategy heuristics inspired by Aaron Fuegi's Tichu article."""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Tuple

from gym_tichu.envs.internals import (
    Card,
    PassAction,
    PlayCombination,
    PlayDog,
    PlayerAction,
    TichuState,
)

from .agents import DefaultGymAgent


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreContribution:
    rule: str
    value: float


@dataclass(frozen=True)
class ScoredAction:
    action: PlayerAction
    contributions: Tuple[ScoreContribution, ...]

    @property
    def total(self):
        return sum(contribution.value for contribution in self.contributions)


def _card_keep_value(card: Card, rank_counts, bomb_cards) -> float:
    special_values = {
        Card.DRAGON: 100,
        Card.PHOENIX: 95,
        Card.MAHJONG: 70,
        Card.DOG: 55,
    }
    value = special_values.get(card, card.card_height)
    value += max(0, rank_counts[card.rank] - 1) * 25
    if card in bomb_cards:
        value += 100
    return value


def fuegi_trading_strategy(state: TichuState, player: int):
    """Trade two awkward cards to opponents and support the partner."""
    hand = state.handcards[player]
    rank_counts = Counter(card.rank for card in hand)
    bomb_cards = {
        card
        for combination in hand.possible_combinations()
        if combination.is_bomb()
        for card in combination.cards
    }
    ranked_cards = sorted(
        hand,
        key=lambda card: (_card_keep_value(card, rank_counts, bomb_cards), card.number),
    )

    protected_specials = {Card.DRAGON, Card.PHOENIX, Card.MAHJONG, Card.DOG}
    opponent_candidates = [
        card for card in ranked_cards if card not in protected_specials
    ]
    if len(opponent_candidates) < 2:
        opponent_candidates = ranked_cards
    opponent_cards = opponent_candidates[:2]
    remaining_cards = [card for card in ranked_cards if card not in opponent_cards]
    strength = (
        2 * sum(card in (Card.DRAGON, Card.PHOENIX) for card in hand)
        + sum(card.rank.name == "A" for card in hand)
        + 2 * bool(bomb_cards)
    )
    partner_card = remaining_cards[0] if strength >= 4 else remaining_cards[-1]

    # TichuGame maps the return positions to left opponent, partner, right opponent.
    return opponent_cards[0], partner_card, opponent_cards[1]


def give_dragon_to_opponent_with_more_cards(state: TichuState, player: int) -> int:
    left = (player - 1) % 4
    right = (player + 1) % 4
    return max((left, right), key=lambda position: (len(state.handcards[position]), position))


class FuegiActionScorer:
    """Assign named strategic contributions to each legal play action."""

    def score(self, state: TichuState, action: PlayerAction) -> ScoredAction:
        contributions = []
        player = state.player_pos
        partner = (player + 2) % 4
        trick_winner = state.trick_on_table.winner
        partner_winning = trick_winner == partner

        if isinstance(action, PassAction):
            contributions.append(ScoreContribution("base_pass", 0))
            if partner_winning:
                contributions.append(ScoreContribution("do_not_overtake_partner", 120))
            if self._opponent_close_to_finishing(state, player):
                contributions.append(ScoreContribution("stop_finishing_opponent", -40))
            return ScoredAction(action, tuple(contributions))

        if not isinstance(action, PlayCombination):
            return ScoredAction(action, (ScoreContribution("neutral_action", 0),))

        combination = action.combination
        cards = set(combination.cards)
        hand_size = len(state.handcards[player])

        contributions.append(ScoreContribution("cards_removed", 12 * len(combination)))
        contributions.append(ScoreContribution("use_lowest_sufficient_group", -combination.height))

        if state.trick_on_table.is_empty():
            contributions.append(ScoreContribution("lead_low", 2 * (15 - combination.height)))

        if len(combination) == hand_size:
            contributions.append(ScoreContribution("finish_hand", 1000))

        if partner_winning:
            contributions.append(ScoreContribution("do_not_overtake_partner", -120))

        if combination.is_bomb() and len(combination) != hand_size:
            contributions.append(ScoreContribution("preserve_bomb", -80))

        if Card.DRAGON in cards and hand_size > 3:
            contributions.append(ScoreContribution("preserve_dragon", -60))

        if Card.PHOENIX in cards and hand_size > 3:
            contributions.append(ScoreContribution("preserve_phoenix", -45))

        if isinstance(action, PlayDog):
            contributions.append(ScoreContribution("use_dog_early", 45 if hand_size > 3 else 0))

        if self._opponent_close_to_finishing(state, player):
            contributions.append(ScoreContribution("stop_finishing_opponent", 20))

        return ScoredAction(action, tuple(contributions))

    @staticmethod
    def _opponent_close_to_finishing(state: TichuState, player: int) -> bool:
        opponents = ((player + 1) % 4, (player - 1) % 4)
        return any(
            opponent not in state.ranking and 0 < len(state.handcards[opponent]) <= 2
            for opponent in opponents
        )


class FuegiHeuristicAgent(DefaultGymAgent):
    """Choose the legal action with the highest explainable heuristic score."""

    def __init__(self, scorer=None):
        super().__init__(
            trade=fuegi_trading_strategy,
            give_dragon_away=give_dragon_to_opponent_with_more_cards,
        )
        self.scorer = scorer or FuegiActionScorer()
        self.last_scored_actions = ()

    @property
    def info(self):
        return "FuegiHeuristicAgent, explainable rule-based play and trading"

    def action(self, state: TichuState) -> PlayerAction:
        actions = sorted(state.possible_actions_list, key=repr)
        self.last_scored_actions = tuple(
            self.scorer.score(state, action) for action in actions
        )
        selected = max(self.last_scored_actions, key=lambda scored: scored.total)
        logger.debug(
            "Fuegi action %s score=%s contributions=%s",
            selected.action,
            selected.total,
            selected.contributions,
        )
        return selected.action

    def explain_action(self, state: TichuState, action: PlayerAction):
        return self.scorer.score(state, action)
