from scraper.brettspielwelt_replay import card_from_token
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
