from pathlib import Path

from gym_agents.combination_weights import CombinationWeights
from gym_agents.mcts import make_default_ismctsearch
from gym_tichu.envs.internals.cards import Card


def test_packaged_combination_weights_can_be_loaded_without_scraper():
    weights_file = Path(__file__).parents[1] / "gym_agents" / "gcombweights.pkl"

    weights = CombinationWeights.weights_from_file(weights_file)

    assert weights


def test_default_mcts_returns_a_legal_action(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    search = make_default_ismctsearch(name="RegressionTestMcts")

    action = search.search(
        root_state=state,
        observer_id=state.player_pos,
        iterations=2,
        cheat=True,
        max_time=1,
    )

    assert action in state.possible_actions_set
