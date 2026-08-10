import pytest

from gym_tichu.envs.internals.cards import (
    Card,
    CardSet,
    Combination,
    FullHouse,
    Pair,
    Single,
    SquareBomb,
    Straight,
    StraightBomb,
    Trio,
)


def test_single_uses_card_height():
    single = Single(Card.A_JADE)

    assert single.height == 14
    assert single.cards == {Card.A_JADE}


def test_dragon_beats_ace_and_dog_cannot_beat_two():
    assert Single(Card.DRAGON).can_be_played_on(Single(Card.A_JADE))
    assert not Single(Card.DOG).can_be_played_on(Single(Card.TWO_JADE))


def test_pair_requires_matching_ranks():
    pair = Pair(Card.FIVE_JADE, Card.FIVE_SWORD)

    assert pair.height == 5
    assert len(pair.cards) == 2

    with pytest.raises(ValueError):
        Pair(Card.FIVE_JADE, Card.SIX_SWORD)


def test_phoenix_can_complete_pair():
    pair = Pair(Card.PHOENIX, Card.NINE_JADE)

    assert pair.height == 9
    assert pair.contains_phoenix()


def test_phoenix_single_is_half_a_rank_above_the_previous_single():
    responses = list(
        CardSet({Card.PHOENIX}).possible_combinations(
            played_on=Single(Card.A_JADE)
        )
    )

    assert len(responses) == 1
    assert responses[0].cards == {Card.PHOENIX}
    assert responses[0].height == 14.5


def test_trio_requires_matching_ranks():
    trio = Trio(Card.SEVEN_JADE, Card.SEVEN_SWORD, Card.SEVEN_PAGODA)

    assert trio.height == 7

    with pytest.raises(ValueError):
        Trio(Card.SEVEN_JADE, Card.SEVEN_SWORD, Card.EIGHT_PAGODA)


def test_full_house_height_is_its_trio_height():
    full_house = FullHouse.from_cards(
        {
            Card.FOUR_JADE,
            Card.FOUR_SWORD,
            Card.NINE_JADE,
            Card.NINE_SWORD,
            Card.NINE_PAGODA,
        }
    )

    assert full_house.height == 9
    assert len(full_house.cards) == 5


def test_higher_full_house_is_generated_as_a_response():
    played = FullHouse.from_cards(
        {
            Card.TWO_JADE,
            Card.TWO_SWORD,
            Card.NINE_JADE,
            Card.NINE_SWORD,
            Card.NINE_PAGODA,
        }
    )
    hand = CardSet(
        {
            Card.SEVEN_JADE,
            Card.SEVEN_SWORD,
            Card.K_JADE,
            Card.K_SWORD,
            Card.K_PAGODA,
        }
    )

    responses = list(hand.possible_combinations(played_on=played))

    assert any(
        isinstance(combination, FullHouse) and combination.height == 13
        for combination in responses
    )


def test_straight_requires_five_consecutive_ranks():
    straight = Straight(
        {
            Card.TWO_JADE,
            Card.THREE_SWORD,
            Card.FOUR_PAGODA,
            Card.FIVE_JADE,
            Card.SIX_SWORD,
        }
    )

    assert straight.lowest.value == 2
    assert straight.highest.value == 6

    with pytest.raises(ValueError):
        Straight(
            {
                Card.TWO_JADE,
                Card.THREE_SWORD,
                Card.FIVE_PAGODA,
                Card.SIX_JADE,
                Card.SEVEN_SWORD,
            }
        )


def test_phoenix_can_fill_gap_in_straight():
    straight = Straight(
        {
            Card.TWO_JADE,
            Card.THREE_SWORD,
            Card.PHOENIX,
            Card.FIVE_JADE,
            Card.SIX_SWORD,
        },
        phoenix_as=Card.FOUR_SWORD,
    )

    assert straight.contains_phoenix()
    assert straight.lowest.value == 2
    assert straight.highest.value == 6


def test_square_bomb_requires_four_cards_of_same_rank():
    bomb = SquareBomb(
        Card.TEN_JADE,
        Card.TEN_SWORD,
        Card.TEN_PAGODA,
        Card.TEN_HOUSE,
    )

    assert bomb.is_bomb()
    assert bomb.rank.value == 10

    with pytest.raises(ValueError):
        SquareBomb(
            Card.TEN_JADE,
            Card.TEN_SWORD,
            Card.TEN_PAGODA,
            Card.J_HOUSE,
        )


def test_straight_bomb_requires_one_suit():
    bomb = StraightBomb.from_cards(
        Card.TWO_JADE,
        Card.THREE_JADE,
        Card.FOUR_JADE,
        Card.FIVE_JADE,
        Card.SIX_JADE,
    )

    assert bomb.is_bomb()
    assert bomb.highest.value == 6

    mixed_suit_straight = Straight(
        {
            Card.TWO_JADE,
            Card.THREE_SWORD,
            Card.FOUR_JADE,
            Card.FIVE_JADE,
            Card.SIX_JADE,
        }
    )
    with pytest.raises(AssertionError):
        StraightBomb(mixed_suit_straight)


def test_combination_factory_identifies_common_combinations():
    pair = Combination.make({Card.Q_JADE, Card.Q_SWORD})
    bomb = Combination.make(
        {
            Card.K_JADE,
            Card.K_SWORD,
            Card.K_PAGODA,
            Card.K_HOUSE,
        }
    )

    assert isinstance(pair, Pair)
    assert isinstance(bomb, SquareBomb)


def test_higher_pair_beats_lower_pair_but_not_single():
    low_pair = Pair(Card.THREE_JADE, Card.THREE_SWORD)
    high_pair = Pair(Card.FOUR_JADE, Card.FOUR_SWORD)

    assert high_pair.can_be_played_on(low_pair)
    assert not low_pair.can_be_played_on(high_pair)
    assert not high_pair.can_be_played_on(Single(Card.THREE_PAGODA))
