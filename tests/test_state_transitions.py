import pytest

from gym_tichu.envs.internals.actions import (
    GiveDragonAwayAction,
    PassAction,
    PassBombAction,
    PlayBomb,
    PlayDog,
    PlayFirst,
    PlayCombination,
    Trick,
    WishAction,
)
from gym_tichu.envs.internals.cards import (
    Card,
    CardRank,
    Single,
    SquareBomb,
    Straight,
)
from gym_tichu.envs.internals.error import IllegalActionError


def test_play_removes_card_and_advances_turn(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
            {Card.SIX_JADE},
        ]
    )
    action = PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))

    next_state = state.next_state(action)

    assert next_state.handcards[0] == {Card.THREE_JADE}
    assert next_state.player_pos == 1
    assert next_state.trick_on_table.last_combination == Single(Card.TWO_JADE)
    assert state.handcards[0] == {Card.TWO_JADE, Card.THREE_JADE}


def test_playing_last_card_adds_player_to_ranking(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )

    next_state = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))
    )

    assert not next_state.handcards[0]
    assert next_state.ranking == (0,)
    assert next_state.player_pos == 1


def test_illegal_play_is_rejected_without_changing_state(state_factory):
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )
    illegal_action = PlayFirst(player_pos=0, combination=Single(Card.A_JADE))

    with pytest.raises(IllegalActionError):
        state.next_state(illegal_action)

    assert state.handcards[0] == {Card.TWO_JADE}
    assert state.trick_on_table.is_empty()


def test_dog_passes_lead_to_partner(state_factory):
    state = state_factory(
        [
            {Card.DOG, Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ]
    )

    next_state = state.next_state(PlayDog(player_pos=0))

    assert next_state.player_pos == 2
    assert Card.DOG not in next_state.handcards[0]


def test_three_passes_finish_trick_and_return_lead(state_factory):
    opening_trick = Trick(
        [PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))]
    )
    state = state_factory(
        [
            {Card.SIX_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=1,
        trick_on_table=opening_trick,
    )

    after_first_pass = state.next_state(PassAction(player_pos=1))
    after_second_pass = after_first_pass.next_state(PassAction(player_pos=2))
    after_third_pass = after_second_pass.next_state(PassAction(player_pos=3))

    assert not after_first_pass.trick_on_table.is_finished()
    assert not after_second_pass.trick_on_table.is_finished()
    assert after_third_pass.trick_on_table.is_finished()
    assert after_third_pass.player_pos == 0


def test_mahjong_requires_wish_before_play_continues(state_factory):
    mahjong_trick = Trick(
        [PlayFirst(player_pos=0, combination=Single(Card.MAHJONG))]
    )
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=1,
        trick_on_table=mahjong_trick,
    )

    assert state.possible_actions_list
    assert all(isinstance(action, WishAction) for action in state.possible_actions_list)
    assert {action.player_pos for action in state.possible_actions_list} == {0}

    next_state = state.next_state(WishAction(player_pos=0, wish=CardRank.FIVE))

    assert next_state.wish is CardRank.FIVE
    assert next_state.player_pos == 1
    assert next_state.history.wished()


def test_wish_must_be_fulfilled_and_is_then_cleared(state_factory):
    opening_trick = Trick(
        [PlayFirst(player_pos=0, combination=Single(Card.THREE_JADE))]
    )
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.FIVE_JADE, Card.SIX_JADE},
            {Card.SEVEN_JADE},
            {Card.EIGHT_JADE},
        ],
        player_pos=1,
        trick_on_table=opening_trick,
        wish=CardRank.FIVE,
    )

    assert not any(isinstance(action, PassAction) for action in state.possible_actions_list)
    playable_actions = [
        action
        for action in state.possible_actions_list
        if isinstance(action, PlayCombination)
    ]
    assert playable_actions
    assert all(
        action.combination.contains_cardrank(CardRank.FIVE)
        for action in playable_actions
    )

    next_state = state.next_state(
        PlayCombination(player_pos=1, combination=Single(Card.FIVE_JADE))
    )

    assert next_state.wish is None
    assert Card.FIVE_JADE not in next_state.handcards[1]


def test_dragon_trick_is_given_to_selected_opponent(state_factory):
    dragon_trick = Trick(
        [PlayFirst(player_pos=0, combination=Single(Card.DRAGON))]
    ).finish()
    state = state_factory(
        [
            {Card.TWO_JADE},
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
            {Card.FIVE_JADE},
        ],
        player_pos=0,
        trick_on_table=dragon_trick,
    )
    action = GiveDragonAwayAction(
        player_from=0,
        player_to=1,
        trick=dragon_trick,
    )

    next_state = state.next_state(action)

    assert next_state.trick_on_table.is_empty()
    assert next_state.won_tricks[1] == (dragon_trick,)
    assert next_state.player_pos == 0


def test_final_dragon_trick_is_still_given_to_selected_opponent(state_factory):
    dragon_trick = Trick(
        [PlayFirst(player_pos=0, combination=Single(Card.DRAGON))]
    ).finish()
    state = state_factory(
        [
            set(),
            set(),
            set(),
            {Card.FIVE_JADE},
        ],
        player_pos=3,
        ranking=(1, 2, 0),
        trick_on_table=dragon_trick,
    )

    actions = state.possible_actions_list

    assert len(actions) == 2
    assert all(isinstance(action, GiveDragonAwayAction) for action in actions)
    give_to_three = next(action for action in actions if action.to == 3)
    terminal = state.next_state(give_to_three)
    assert terminal.is_terminal()
    assert terminal.won_tricks[3] == (dragon_trick,)


def test_double_win_ends_round_and_scores_200_for_team(state_factory):
    state = state_factory(
        [
            set(),
            {Card.THREE_JADE},
            set(),
            {Card.FIVE_JADE},
        ],
        ranking=(0, 2),
    )

    assert state.is_terminal()
    assert state.is_double_win()
    assert state.count_points() == (200, 0, 200, 0)


def test_out_of_turn_square_bomb_interrupts_normal_player(state_factory):
    fours = {
        Card.FOUR_JADE,
        Card.FOUR_HOUSE,
        Card.FOUR_SWORD,
        Card.FOUR_PAGODA,
    }
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.SIX_JADE},
            fours | {Card.SEVEN_JADE},
            {Card.EIGHT_JADE},
        ]
    )

    after_play = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))
    )

    assert after_play.player_pos == 2
    assert after_play.bomb_window == (2,)
    bomb_action = next(
        action
        for action in after_play.possible_actions_list
        if isinstance(action, PlayBomb)
    )
    after_bomb = after_play.next_state(bomb_action)

    assert after_bomb.trick_on_table.last_combination == SquareBomb(*fours)
    assert not fours.intersection(after_bomb.handcards[2])
    assert after_bomb.player_pos == 3


def test_declining_async_bomb_restores_interrupted_turn(state_factory):
    fours = {
        Card.FOUR_JADE,
        Card.FOUR_HOUSE,
        Card.FOUR_SWORD,
        Card.FOUR_PAGODA,
    }
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.SIX_JADE},
            fours | {Card.SEVEN_JADE},
            {Card.EIGHT_JADE},
        ]
    )
    bomb_window = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))
    )

    resumed = bomb_window.next_state(PassBombAction(player_pos=2))

    assert resumed.bomb_window == ()
    assert resumed.player_pos == 1
    assert any(isinstance(action, PassAction) for action in resumed.possible_actions_list)


def test_higher_async_bomb_can_rebomb(state_factory):
    fours = {
        Card.FOUR_JADE,
        Card.FOUR_HOUSE,
        Card.FOUR_SWORD,
        Card.FOUR_PAGODA,
    }
    fives = {
        Card.FIVE_JADE,
        Card.FIVE_HOUSE,
        Card.FIVE_SWORD,
        Card.FIVE_PAGODA,
    }
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.SIX_JADE},
            fours | {Card.SEVEN_JADE},
            fives | {Card.EIGHT_JADE},
        ]
    )
    first_window = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))
    )
    first_bomb = next(
        action
        for action in first_window.possible_actions_list
        if isinstance(action, PlayBomb) and action.player_pos == 2
    )
    second_window = first_window.next_state(first_bomb)
    second_bomb = next(
        action
        for action in second_window.possible_actions_list
        if isinstance(action, PlayBomb) and action.player_pos == 3
    )

    after_rebomb = second_window.next_state(second_bomb)

    assert after_rebomb.trick_on_table.last_combination == SquareBomb(*fives)
    assert after_rebomb.player_pos == 0


def test_bomb_may_be_played_after_three_normal_passes(state_factory):
    fours = {
        Card.FOUR_JADE,
        Card.FOUR_HOUSE,
        Card.FOUR_SWORD,
        Card.FOUR_PAGODA,
    }
    state = state_factory(
        [
            {Card.TWO_JADE, Card.THREE_JADE},
            {Card.SIX_JADE},
            fours | {Card.SEVEN_JADE},
            {Card.EIGHT_JADE},
        ]
    )
    state = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TWO_JADE))
    )
    state = state.next_state(PassBombAction(player_pos=2))

    state = state.next_state(PassAction(player_pos=1))
    state = state.next_state(PassBombAction(player_pos=2))
    state = state.next_state(PassAction(player_pos=2))
    state = state.next_state(PassBombAction(player_pos=2))
    state = state.next_state(PassAction(player_pos=3))

    assert not state.trick_on_table.is_finished()
    assert state.bomb_trick_finish is True
    assert any(isinstance(action, PlayBomb) for action in state.possible_actions_list)

    finished = state.next_state(PassBombAction(player_pos=2))

    assert finished.trick_on_table.is_finished()
    assert finished.player_pos == 0


def test_late_bomb_starts_new_trick_without_stealing_completed_points(state_factory):
    fours = {
        Card.FOUR_JADE,
        Card.FOUR_HOUSE,
        Card.FOUR_SWORD,
        Card.FOUR_PAGODA,
    }
    state = state_factory(
        [
            {Card.TEN_JADE},
            {Card.SIX_JADE},
            fours | {Card.SEVEN_JADE},
            {Card.EIGHT_JADE},
        ]
    )
    state = state.next_state(
        PlayFirst(player_pos=0, combination=Single(Card.TEN_JADE))
    )
    state = state.next_state(PassBombAction(player_pos=2))
    state = state.next_state(PassAction(player_pos=1))
    state = state.next_state(PassBombAction(player_pos=2))
    state = state.next_state(PassAction(player_pos=2))
    state = state.next_state(PassBombAction(player_pos=2))
    state = state.next_state(PassAction(player_pos=3))

    bomb = next(
        action for action in state.possible_actions_list if isinstance(action, PlayBomb)
    )
    after_bomb = state.next_state(bomb)

    assert state.won_tricks[0][0].points == 10
    assert after_bomb.won_tricks[0][0].points == 10
    assert after_bomb.trick_on_table.last_combination == SquareBomb(*fours)


def test_equal_natural_straight_is_not_mandatory_for_wish(state_factory):
    phoenix_straight = Straight(
        {
            Card.FIVE_HOUSE,
            Card.SIX_HOUSE,
            Card.PHOENIX,
            Card.EIGHT_JADE,
            Card.NINE_SWORD,
        },
        phoenix_as=Card.SEVEN_SWORD,
    )
    state = state_factory(
        [
            {Card.TWO_JADE},
            {
                Card.FIVE_JADE,
                Card.SIX_JADE,
                Card.SEVEN_JADE,
                Card.EIGHT_HOUSE,
                Card.NINE_HOUSE,
            },
            {Card.THREE_JADE},
            {Card.FOUR_JADE},
        ],
        player_pos=1,
        trick_on_table=Trick(
            [PlayFirst(player_pos=0, combination=phoenix_straight)]
        ),
        wish=CardRank.SEVEN,
    )

    assert any(isinstance(action, PassAction) for action in state.possible_actions_list)
    assert any(
        isinstance(action, PlayCombination)
        and action.combination.contains_cardrank(CardRank.SEVEN)
        for action in state.possible_actions_list
    )
