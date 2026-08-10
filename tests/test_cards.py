import pytest

from gym_tichu.envs.internals.cards import Card, CardRank, Deck


def test_full_deck_contains_56_unique_cards():
    deck = Deck()

    assert len(deck) == 56
    assert len(set(deck)) == 56


def test_card_numbers_are_unique_and_cover_expected_range():
    numbers = {card.number for card in Card}

    assert numbers == set(range(56))


def test_full_deck_is_worth_100_points():
    assert sum(card.points for card in Deck()) == 100


def test_special_and_scoring_card_values():
    assert Card.DOG.points == 0
    assert Card.MAHJONG.points == 0
    assert Card.DRAGON.points == 25
    assert Card.PHOENIX.points == -25
    assert Card.FIVE_JADE.points == 5
    assert Card.TEN_JADE.points == 10
    assert Card.K_JADE.points == 10


def test_card_rank_lookup_by_name_and_value():
    assert CardRank.from_name("A") is CardRank.A
    assert CardRank.from_value(14) is CardRank.A
    assert CardRank.from_value(1.5) is CardRank.PHOENIX


def test_deck_splits_into_four_disjoint_hands():
    hands = Deck().split(nbr_piles=4, random_=False)

    assert [len(hand) for hand in hands] == [14, 14, 14, 14]
    assert set().union(*map(set, hands)) == set(Card)


def test_deck_rejects_uneven_split():
    deck = Deck(full=False, cards=list(Card)[:5])

    with pytest.raises(ValueError):
        deck.split(nbr_piles=4)
