from gym_tichu.envs.internals.actions import PlayFirst, Trick
from gym_tichu.envs.internals.cards import Card, Single
from gym_tichu.envs.internals.tichu_state import WonTricks


def scoring_trick(player_pos, card):
    return Trick(
        [PlayFirst(player_pos=player_pos, combination=Single(card))]
    ).finish()


def test_won_trick_points_are_combined_by_team(state_factory):
    won_tricks = WonTricks(
        [scoring_trick(0, Card.TEN_JADE)],
        [scoring_trick(1, Card.FIVE_JADE)],
        [scoring_trick(2, Card.K_JADE)],
        [],
    )
    state = state_factory(
        [set(), set(), set(), {Card.TWO_JADE}],
        ranking=(0, 1, 2),
        won_tricks=won_tricks,
    )

    assert state.is_terminal()
    assert state.count_points() == (20, 5, 20, 5)


def test_last_players_won_tricks_go_to_round_winner(state_factory):
    dragon_trick = scoring_trick(3, Card.DRAGON)
    state = state_factory(
        [set(), set(), set(), {Card.TWO_JADE}],
        ranking=(0, 1, 2),
        won_tricks=WonTricks([], [], [], [dragon_trick]),
    )

    assert state.count_points() == (25, 0, 25, 0)


def test_last_players_hand_points_go_to_opposing_team(state_factory):
    state = state_factory(
        [
            set(),
            set(),
            set(),
            {Card.FIVE_JADE, Card.TEN_JADE},
        ],
        ranking=(0, 1, 2),
    )

    assert state.count_points() == (15, 0, 15, 0)


def test_phoenix_in_last_players_hand_reduces_opponents_score(state_factory):
    state = state_factory(
        [
            set(),
            set(),
            set(),
            {Card.FIVE_JADE, Card.PHOENIX},
        ],
        ranking=(0, 1, 2),
    )

    assert state.count_points() == (-20, 0, -20, 0)


def test_successful_and_failed_tichu_are_scored_per_team(state_factory):
    state = state_factory(
        [set(), set(), set(), {Card.TWO_JADE}],
        ranking=(0, 1, 2),
        announced_tichu=frozenset({0, 1}),
    )

    assert state.count_points() == (100, -100, 100, -100)


def test_successful_and_failed_grand_tichu_are_scored_per_team(state_factory):
    state = state_factory(
        [set(), set(), set(), {Card.TWO_JADE}],
        ranking=(2, 1, 0),
        announced_grand_tichu=frozenset({2, 3}),
    )

    assert not state.is_double_win()
    assert state.count_points() == (200, -200, 200, -200)
