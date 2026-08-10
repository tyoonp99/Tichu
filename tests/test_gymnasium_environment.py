import gymnasium as gym

import gym_tichu  # noqa: F401 - importing registers the environments
from gamemanager import TichuGame
from gym_tichu.envs.internals.tichu_state import (
    AfterTrading,
    BeforeTrading,
    FullCardsState,
    InitialState,
)


class FirstLegalActionAgent:
    def announce_grand_tichu(self, **kwargs):
        return False

    def announce_tichu(self, **kwargs):
        return False

    def trade(self, state, player):
        return list(state.handcards[player])[:3]

    def make_wish(self, **kwargs):
        return None

    def give_dragon_away(self, player, **kwargs):
        return (player + 1) % 4

    def action(self, state):
        return state.possible_actions_list[0]


def test_registered_environment_uses_gymnasium_reset_and_step_api():
    env = gym.make("tichu_multiplayer-v0")

    initial_state, info = env.reset(seed=123)

    assert isinstance(initial_state, InitialState)
    assert info == {}
    assert [len(hand) for hand in initial_state.handcards] == [8, 8, 8, 8]

    full_cards, reward, terminated, truncated, info = env.step(set())

    assert isinstance(full_cards, FullCardsState)
    assert reward == (0, 0, 0, 0)
    assert not terminated
    assert not truncated
    assert info == {}
    assert [len(hand) for hand in full_cards.handcards] == [14, 14, 14, 14]

    before_trading, _, terminated, truncated, _ = env.step(set())
    assert isinstance(before_trading, BeforeTrading)
    assert not terminated
    assert not truncated

    after_trading, _, terminated, truncated, _ = env.step([])
    assert isinstance(after_trading, AfterTrading)
    assert not terminated
    assert not truncated
    assert sum(len(hand) for hand in after_trading.handcards) == 56

    env.close()


def test_multiplayer_environment_can_finish_a_complete_round():
    env = gym.make("tichu_multiplayer-v0")
    state, _ = env.reset(seed=321)

    state, _, _, _, _ = env.step(set())
    state, _, _, _, _ = env.step(set())
    state, _, _, _, _ = env.step([])

    terminated = False
    truncated = False
    reward = (0, 0, 0, 0)

    for _ in range(2_000):
        assert state.possible_actions_list
        state, reward, terminated, truncated, _ = env.step(
            state.possible_actions_list[0]
        )
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert state.is_terminal()
    assert len(state.ranking) >= 2
    assert reward[0] == reward[2]
    assert reward[1] == reward[3]

    env.close()


def test_reset_seed_reproduces_the_same_deal():
    first_env = gym.make("tichu_multiplayer-v0")
    first_initial, _ = first_env.reset(seed=42)
    first_full, _, _, _, _ = first_env.step(set())

    second_env = gym.make("tichu_multiplayer-v0")
    second_initial, _ = second_env.reset(seed=42)
    second_full, _, _, _, _ = second_env.step(set())

    assert first_initial.handcards == second_initial.handcards
    assert first_full.handcards == second_full.handcards

    first_env.close()
    second_env.close()


def test_singleplayer_environment_uses_gymnasium_api():
    env = gym.make("tichu_singleplayer-v0")
    agent = FirstLegalActionAgent()
    env.unwrapped.configure(other_agents=(agent, agent, agent))

    state, info = env.reset(seed=99)

    assert info == {}
    assert state.player_pos == 0

    result = env.step(state.possible_actions_list[0])

    assert len(result) == 5
    next_state, reward, terminated, truncated, info = result
    assert terminated or next_state.player_pos == 0
    assert isinstance(reward, int)
    assert not truncated
    assert isinstance(info, dict)

    env.close()


def test_game_manager_can_finish_a_round_with_simple_agents():
    agents = [FirstLegalActionAgent() for _ in range(4)]
    game = TichuGame(*agents)

    points, history = game._start_round()

    assert len(points) == 2
    assert history.last_state().is_terminal()

    game.env.close()
