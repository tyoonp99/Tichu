import random

import pytest

from gamemanager import TichuGame
from gym_agents import (
    BalancedRandomAgent,
    DQNAgent2L_56x5,
    RandomAgent,
)


def test_random_agents_import_without_legacy_rl_dependencies():
    assert isinstance(RandomAgent(), RandomAgent)
    assert isinstance(BalancedRandomAgent(), BalancedRandomAgent)


def test_legacy_dqn_agent_reports_missing_optional_dependencies():
    with pytest.raises(ImportError, match="legacy DQN agents require"):
        DQNAgent2L_56x5()


@pytest.mark.parametrize("agent_type", [RandomAgent, BalancedRandomAgent])
@pytest.mark.parametrize("seed", range(5))
def test_random_agents_finish_rounds_across_multiple_deals(agent_type, seed):
    random.seed(seed)
    agents = [agent_type() for _ in range(4)]
    game = TichuGame(*agents)

    points, history = game._start_round()

    assert len(points) == 2
    assert history.last_state().is_terminal()

    game.env.close()


@pytest.mark.parametrize("agent_type", [RandomAgent, BalancedRandomAgent])
def test_random_agents_finish_a_game_to_target_score(agent_type):
    random.seed(100)
    agents = [agent_type() for _ in range(4)]
    game = TichuGame(*agents)

    points, histories = game.start_game(target_points=100)

    assert max(points) >= 100
    assert histories
    assert all(history.last_state().is_terminal() for history in histories)

    game.env.close()
