import pytest

from gamemanager import TichuGame
from gym_agents import FuegiHeuristicAgent
from gym_tichu.envs.internals.actions import PassAction, PlayFirst, Trick
from gym_tichu.envs.internals.cards import Card, Single


def test_fuegi_leads_low_from_isolated_single_cards(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )

    action = FuegiHeuristicAgent().action(state)

    assert action.combination == Single(Card.TWO_JADE)


def test_fuegi_does_not_overtake_partner_who_is_winning(state_factory):
    trick = Trick([PlayFirst(player_pos=2, combination=Single(Card.FIVE_JADE))])
    state = state_factory(
        [
            {Card.SIX_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.TWO_JADE},
        ],
        player_pos=0,
        trick_on_table=trick,
    )

    action = FuegiHeuristicAgent().action(state)

    assert isinstance(action, PassAction)


def test_fuegi_prioritizes_finishing_the_hand(state_factory):
    state = state_factory(
        [
            {Card.DRAGON},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )

    action = FuegiHeuristicAgent().action(state)

    assert action.combination == Single(Card.DRAGON)


def test_fuegi_explanation_contains_named_contributions(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    agent = FuegiHeuristicAgent()
    action = agent.action(state)

    explanation = agent.explain_action(state, action)

    assert explanation.total > 0
    assert {item.rule for item in explanation.contributions} >= {
        "cards_removed",
        "lead_low",
    }


def test_fuegi_action_does_not_depend_on_hidden_opponent_cards(state_factory):
    first_state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE, Card.FOUR_JADE},
            {Card.FIVE_JADE, Card.SIX_JADE},
            {Card.SEVEN_JADE, Card.EIGHT_JADE},
        ]
    )
    second_state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.NINE_JADE, Card.TEN_JADE},
            {Card.J_JADE, Card.Q_JADE},
            {Card.K_JADE, Card.DRAGON},
        ]
    )

    first_action = FuegiHeuristicAgent().action(first_state)
    second_action = FuegiHeuristicAgent().action(second_state)

    assert first_action == second_action


def test_fuegi_trading_keeps_special_cards_away_from_opponents(state_factory):
    hand = {
        Card.DRAGON,
        Card.PHOENIX,
        Card.MAHJONG,
        Card.DOG,
        Card.A_JADE,
        Card.A_HOUSE,
        Card.K_JADE,
        Card.Q_JADE,
        Card.J_JADE,
        Card.TEN_JADE,
        Card.NINE_JADE,
        Card.EIGHT_JADE,
        Card.SEVEN_JADE,
        Card.SIX_JADE,
    }
    state = state_factory(
        [hand, {Card.THREE_JADE}, {Card.FOUR_JADE}, {Card.FIVE_JADE}]
    )
    agent = FuegiHeuristicAgent()

    left_opponent, partner, right_opponent = agent.trade(state=state, player=0)

    assert len({left_opponent, partner, right_opponent}) == 3
    assert left_opponent not in {Card.DRAGON, Card.PHOENIX, Card.MAHJONG, Card.DOG}
    assert right_opponent not in {Card.DRAGON, Card.PHOENIX, Card.MAHJONG, Card.DOG}


@pytest.mark.parametrize("seed", range(3))
def test_fuegi_agents_finish_complete_rounds(seed):
    game = TichuGame(*(FuegiHeuristicAgent() for _ in range(4)))
    try:
        points, history = game._start_round(seed=seed)
    finally:
        game.env.close()

    assert len(points) == 2
    assert history.last_state().is_terminal()
