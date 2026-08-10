from scraper.prepare_brettspielwelt_dataset import split_for_game, winning_team


def test_game_split_is_deterministic():
    assert split_for_game(2419834, 0.1) == split_for_game(2419834, 0.1)


def test_winning_team_uses_even_and_odd_seat_teams():
    assert winning_team((120, 80)) == 0
    assert winning_team((45, 155)) == 1
    assert winning_team((100, 100)) is None
