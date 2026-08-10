from scraper.brettspielwelt_parser import (
    final_hands,
    parse_game,
    validate_game,
)


def _suit_cards(suit):
    return [suit + rank for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "B", "D", "K", "A"]]


def sample_log():
    hands = {
        0: ["Ma"] + _suit_cards("R"),
        1: ["Hu"] + _suit_cards("G"),
        2: ["Ph"] + _suit_cards("B"),
        3: ["Dr"] + _suit_cards("S"),
    }
    grand = {player: cards[:8] for player, cards in hands.items()}
    names = ["alpha", "bravo", "charlie", "delta"]
    lines = ["---------------Gr.Tichukarten------------------"]
    lines.extend(
        "({}){} {}".format(player, names[player], " ".join(grand[player]))
        for player in range(4)
    )
    lines.append("---------------Startkarten------------------")
    lines.extend(
        "({}){} {}".format(player, names[player], " ".join(hands[player]))
        for player in range(4)
    )
    lines.extend(
        [
            "Grosses Tichu: (0)alpha",
            "Schupfen:",
            "(0)alpha gibt: bravo: R2 - charlie: R3 - delta: R4 -",
            "(1)bravo gibt: charlie: G2 - delta: G3 - alpha: G4 -",
            "(2)charlie gibt: delta: B2 - alpha: B3 - bravo: B4 -",
            "(3)delta gibt: alpha: S2 - bravo: S3 - charlie: S4 -",
            "BOMBE: (3)delta",
            "---------------Rundenverlauf------------------",
            "Tichu: (1)bravo",
            "(0)alpha: Ma",
            "Wunsch:A",
            "(1)bravo: GA",
            "(2)charlie passt.",
            "Drache an: (2)charlie",
            "Ergebnis: 100 - 0",
        ]
    )
    return "\n".join(lines)


def test_parse_game_preserves_round_information():
    game = parse_game(sample_log())
    round_ = game.rounds[0]

    assert game.players == {0: "alpha", 1: "bravo", 2: "charlie", 3: "delta"}
    assert round_.players == game.players
    assert round_.grand_tichu == {0}
    assert round_.tichu == {1}
    assert round_.bomb_holders == {3}
    assert len(round_.trades) == 12
    assert round_.events[-1].kind == "dragon_to"
    assert round_.scores == (100, 0)


def test_final_hands_applies_exchanges():
    round_ = parse_game(sample_log()).rounds[0]
    hands = final_hands(round_)

    assert "R2" not in hands[0]
    assert {"G4", "B3", "S2"}.issubset(hands[0])
    assert all(len(hand) == 14 for hand in hands.values())


def test_validate_game_accepts_structurally_complete_sample():
    game = parse_game(sample_log())
    assert validate_game(game) == []


def test_player_replacement_at_trade_time_is_assigned_by_seat():
    text = sample_log().replace(
        "(3)delta gibt:", "(3)replacement gibt:"
    ).replace("delta:", "replacement:")
    game = parse_game(text)

    assert game.rounds[0].players[3] == "replacement"
    assert len(game.rounds[0].trades) == 12


def test_multiple_bomb_notices_are_preserved():
    text = sample_log().replace("BOMBE: (3)delta", "BOMBE: (1)bravo (3)delta")
    round_ = parse_game(text).rounds[0]

    assert round_.bomb_holders == {1, 3}
