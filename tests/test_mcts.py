from pathlib import Path

from gym_agents.combination_weights import CombinationWeights
from gym_agents.mcts import make_default_ismctsearch, make_fuegi_ismctsearch
from gym_tichu.envs.internals.actions import PassAction, PlayFirst, Trick
from gym_tichu.envs.internals.cards import Card, Single


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
    assert search.last_search_metrics.completed_iterations == 2
    assert search.last_search_metrics.reached_iteration_limit
    assert not search.last_search_metrics.reached_time_limit


def test_mcts_always_completes_one_iteration_when_time_budget_is_exhausted(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    search = make_default_ismctsearch(name="ZeroTimeBudgetMcts")

    action = search.search(
        root_state=state,
        observer_id=state.player_pos,
        iterations=10,
        cheat=False,
        max_time=0,
    )

    assert action in state.possible_actions_set
    assert search.last_search_metrics.completed_iterations == 1
    assert not search.last_search_metrics.reached_iteration_limit
    assert search.last_search_metrics.reached_time_limit


def test_fuegi_mcts_evaluates_the_final_team_point_difference(state_factory):
    state = state_factory(
        [set(), {Card.THREE_JADE}, set(), {Card.FIVE_JADE}],
        ranking=(0, 2),
    )
    search = make_fuegi_ismctsearch(name="TeamScoreMcts")

    reward = search.evaluate_state(state)

    assert reward == (1, -1, 1, -1)


def test_fuegi_rollout_supports_the_partner(state_factory):
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
    search = make_fuegi_ismctsearch(name="PartnerSupportMcts")

    action = search.rollout_action(state)

    assert isinstance(action, PassAction)


def test_fuegi_mcts_returns_a_legal_action(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    search = make_fuegi_ismctsearch(name="FuegiMctsRegression")

    action = search.search(
        root_state=state,
        observer_id=state.player_pos,
        iterations=2,
        cheat=True,
        max_time=1,
    )

    assert action in state.possible_actions_set


def test_fuegi_mcts_uses_heuristic_priority_with_one_iteration(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.A_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    search = make_fuegi_ismctsearch(name="FuegiPriorityMcts")

    action = search.search(
        root_state=state,
        observer_id=state.player_pos,
        iterations=1,
        cheat=True,
        max_time=1,
    )

    assert action.combination == Single(Card.TWO_JADE)
