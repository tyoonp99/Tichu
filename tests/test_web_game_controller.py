from gym_tichu.envs.internals import wishable_card_ranks

from web_game.controller import WebGame, WebGameError, automatic_wish


class FirstLegalAgent:
    def action(self, state):
        return state.possible_actions_list[0]

    def give_dragon_away(self, state, player):
        return (player + 1) % 4


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def advance_to_human_turn(game, clock, player=0):
    for _ in range(20):
        game.view_for(player)
        if game.waiting_player() == player:
            return
        clock.now += 1.0
    raise AssertionError("human turn was not reached")


def test_web_game_exposes_only_current_human_hand_and_legal_actions():
    game = WebGame(ai_factory=FirstLegalAgent, seed=17)
    viewer = game.waiting_player()

    assert viewer in {0, 2}
    state = game.view_for(viewer)
    assert len(state["hand"]) <= 14
    assert state["waiting_player"] == viewer
    assert state["legal_actions"]
    assert "hands" not in state


def test_web_game_rejects_action_from_non_current_human():
    game = WebGame(ai_factory=FirstLegalAgent, seed=17)
    viewer = game.waiting_player()
    other_human = (viewer + 2) % 4
    action_id = game.view_for(viewer)["legal_actions"][0]["id"]

    try:
        game.play(other_human, action_id)
    except WebGameError as error:
        assert "차례" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-current human action was accepted")


def test_web_game_single_player_mode_uses_three_ai_seats():
    clock = FakeClock()
    game = WebGame(
        ai_factory=FirstLegalAgent,
        seed=17,
        human_seats=frozenset({0}),
        clock=clock,
    )

    advance_to_human_turn(game, clock)
    assert game.waiting_player() == 0
    assert set(game.agents) == {1, 2, 3}
    assert game.view_for(0)["partner"] == 2


def test_web_game_keeps_a_play_event_visible_before_advancing():
    clock = FakeClock()
    game = WebGame(
        ai_factory=FirstLegalAgent,
        seed=17,
        human_seats=frozenset({0}),
        clock=clock,
    )
    advance_to_human_turn(game, clock)
    view = game.view_for(0)
    action_id = next(
        action["id"]
        for action in view["legal_actions"]
        if action["cards"] and "1" not in action["cards"]
    )

    game.play(0, action_id)
    after_play = game.view_for(0)

    assert after_play["last_event"]["cards"]
    assert after_play["animation_remaining"] > 0
    assert after_play["legal_actions"] == []
    assert after_play["table_player"] == after_play["last_event"]["actor"]
    assert "bomb_remaining" not in after_play
    assert game.bomb_deadline == clock.now + game.bomb_window_seconds
    assert after_play["reaction_remaining"] == game.bomb_window_seconds
    assert isinstance(after_play["table_points"], int)


def test_automatic_wish_is_always_a_real_wishable_rank():
    clock = FakeClock()
    game = WebGame(
        ai_factory=FirstLegalAgent,
        seed=17,
        human_seats=frozenset({0}),
        clock=clock,
    )

    wish = automatic_wish(game.state, 1)

    assert wish in wishable_card_ranks
