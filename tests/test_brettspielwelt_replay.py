from scraper.brettspielwelt_replay import (
    ReplayResult,
    _best_branch_result,
    _prepare_for_event,
    card_from_token,
)
from gym_tichu.envs.internals.actions import (
    PassBombAction,
    PlayCombination,
    Trick,
)
from gym_tichu.envs.internals.cards import Card, Single


def test_card_tokens_map_to_engine_cards():
    assert card_from_token("Ph") is Card.PHOENIX
    assert card_from_token("Dr") is Card.DRAGON
    assert card_from_token("Ma") is Card.MAHJONG
    assert card_from_token("Hu") is Card.DOG
    assert card_from_token("R2") is Card.TWO_HOUSE
    assert card_from_token("GA") is Card.A_JADE
    assert card_from_token("BB") is Card.J_PAGODA
    assert card_from_token("SD") is Card.Q_SWORD


def test_ambiguous_phoenix_replay_prefers_branch_matching_final_score():
    early_failure = ReplayResult(status="illegal_pass", event_index=9)
    later_failure = ReplayResult(status="illegal_play", event_index=20)
    score_mismatch = ReplayResult(status="score_mismatch")
    exact = ReplayResult(status="ok")

    assert _best_branch_result([early_failure, later_failure]) is later_failure
    assert _best_branch_result([early_failure, score_mismatch]) is score_mismatch
    assert _best_branch_result([score_mismatch, exact]) is exact


def test_replay_can_capture_bomb_decline_inferred_from_log_omission(
    state_factory,
):
    table_play = PlayCombination(3, Single(Card.THREE_JADE))
    state = state_factory(
        [
            {
                Card.TWO_JADE,
                Card.TWO_HOUSE,
                Card.TWO_PAGODA,
                Card.TWO_SWORD,
            },
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
            {Card.THREE_JADE},
        ],
        player_pos=0,
        trick_on_table=Trick((table_play,)),
        bomb_window=(0,),
        bomb_resume_player=1,
    )
    inferred = []

    resumed = _prepare_for_event(state, inferred_decisions=inferred)

    assert len(inferred) == 1
    assert inferred[0][0] is state
    assert inferred[0][1] is None
    assert isinstance(inferred[0][2], PassBombAction)
    assert resumed.bomb_window == ()
    assert resumed.player_pos == 1
