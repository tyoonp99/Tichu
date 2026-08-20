import pytest

from gym_tichu.envs.internals.actions import Trick
from gym_tichu.envs.internals.tichu_state import (
    HandCards,
    History,
    TichuState,
    WonTricks,
)


@pytest.fixture
def state_factory():
    def make_state(
        hands,
        *,
        player_pos=0,
        trick_on_table=None,
        wish=None,
        ranking=(),
        announced_tichu=frozenset(),
        announced_grand_tichu=frozenset(),
        history=None,
        won_tricks=None,
        allow_tichu=False,
        allow_wish=True,
        bomb_window=(),
        bomb_resume_player=None,
        bomb_trick_finish=False,
    ):
        return TichuState(
            player_pos=player_pos,
            handcards=HandCards(*hands),
            won_tricks=won_tricks or WonTricks(),
            trick_on_table=trick_on_table or Trick(),
            wish=wish,
            ranking=ranking,
            announced_tichu=announced_tichu,
            announced_grand_tichu=announced_grand_tichu,
            history=history or History(),
            bomb_window=bomb_window,
            bomb_resume_player=bomb_resume_player,
            bomb_trick_finish=bomb_trick_finish,
            allow_tichu=allow_tichu,
            allow_wish=allow_wish,
        )

    return make_state
