from scraper.brettspielwelt_replay import (
    ReplayResult,
    _best_branch_result,
    card_from_token,
)
from gym_tichu.envs.internals.cards import Card


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
