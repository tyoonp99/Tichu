from gym_tichu.envs.internals.cards import (
    Card,
    CardRank,
    CardSet,
    Pair,
    PairSteps,
    Single,
    SquareBomb,
    StraightBomb,
)


def test_small_hand_generates_singles_and_pair_without_duplicates():
    hand = CardSet(
        {
            Card.THREE_JADE,
            Card.THREE_SWORD,
            Card.FOUR_JADE,
        }
    )

    combinations = list(hand.possible_combinations())

    assert len(combinations) == len(set(combinations))
    assert len([comb for comb in combinations if isinstance(comb, Single)]) == 2
    assert len([comb for comb in combinations if isinstance(comb, Pair)]) == 1
    assert {comb.height for comb in combinations if isinstance(comb, Single)} == {3, 4}


def test_every_generated_combination_uses_only_cards_from_hand():
    hand = CardSet(
        {
            Card.TWO_JADE,
            Card.THREE_JADE,
            Card.FOUR_JADE,
            Card.FIVE_JADE,
            Card.SIX_JADE,
            Card.SIX_SWORD,
            Card.SEVEN_JADE,
            Card.SEVEN_SWORD,
            Card.PHOENIX,
        }
    )

    combinations = list(hand.possible_combinations())

    assert combinations
    assert all(combination.cards.issubset(hand) for combination in combinations)
    assert len(combinations) == len(set(combinations))


def test_played_single_filters_out_lower_singles():
    hand = CardSet(
        {
            Card.THREE_JADE,
            Card.FIVE_JADE,
            Card.A_JADE,
            Card.DRAGON,
            Card.DOG,
        }
    )

    combinations = list(
        hand.possible_combinations(played_on=Single(Card.FOUR_JADE))
    )
    singles = [comb for comb in combinations if isinstance(comb, Single)]

    assert {single.card for single in singles} == {
        Card.FIVE_JADE,
        Card.A_JADE,
        Card.DRAGON,
    }
    assert all(single.height > 4 for single in singles)


def test_played_pair_allows_only_higher_pairs():
    hand = CardSet(
        {
            Card.THREE_JADE,
            Card.THREE_SWORD,
            Card.FIVE_JADE,
            Card.FIVE_SWORD,
        }
    )
    played_pair = Pair(Card.FOUR_JADE, Card.FOUR_SWORD)

    combinations = list(hand.possible_combinations(played_on=played_pair))

    assert len(combinations) == 1
    assert isinstance(combinations[0], Pair)
    assert combinations[0].height == 5


def test_contains_rank_filters_every_generated_combination():
    hand = CardSet(
        {
            Card.FIVE_JADE,
            Card.FIVE_SWORD,
            Card.SIX_JADE,
            Card.SIX_SWORD,
            Card.SEVEN_JADE,
        }
    )

    combinations = list(
        hand.possible_combinations(contains_rank=CardRank.FIVE)
    )

    assert combinations
    assert all(combination.fulfills_wish(CardRank.FIVE) for combination in combinations)
    assert any(isinstance(combination, Single) for combination in combinations)
    assert any(isinstance(combination, Pair) for combination in combinations)
    assert any(isinstance(combination, PairSteps) for combination in combinations)


def test_nothing_can_be_played_on_dog():
    hand = CardSet(
        {
            Card.A_JADE,
            Card.TEN_JADE,
            Card.TEN_SWORD,
            Card.TEN_PAGODA,
            Card.TEN_HOUSE,
        }
    )

    assert list(hand.possible_combinations(played_on=Single(Card.DOG))) == []


def test_only_bomb_can_be_played_on_dragon():
    hand = CardSet(
        {
            Card.A_JADE,
            Card.TEN_JADE,
            Card.TEN_SWORD,
            Card.TEN_PAGODA,
            Card.TEN_HOUSE,
        }
    )

    combinations = list(
        hand.possible_combinations(played_on=Single(Card.DRAGON))
    )

    assert len(combinations) == 1
    assert isinstance(combinations[0], SquareBomb)


def test_square_and_straight_bombs_are_both_generated():
    hand = CardSet(
        {
            Card.TWO_JADE,
            Card.THREE_JADE,
            Card.FOUR_JADE,
            Card.FIVE_JADE,
            Card.SIX_JADE,
            Card.TEN_JADE,
            Card.TEN_SWORD,
            Card.TEN_PAGODA,
            Card.TEN_HOUSE,
        }
    )

    bombs = [
        combination
        for combination in hand.possible_combinations()
        if combination.is_bomb()
    ]

    assert any(isinstance(bomb, SquareBomb) for bomb in bombs)
    assert any(isinstance(bomb, StraightBomb) for bomb in bombs)
