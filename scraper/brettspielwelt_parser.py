"""Parser and structural validation for Brettspielwelt ``.tch`` logs.

This module is a clean implementation based on the public text format.  It
keeps the original card tokens so raw records can be audited before they are
converted to the engine's card objects.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


GRAND_HEADER = "---------------Gr.Tichukarten------------------"
START_HEADER = "---------------Startkarten------------------"
PLAY_HEADER = "---------------Rundenverlauf------------------"
CARD_TOKEN_RE = re.compile(r"^(?:Ph|Dr|Ma|Hu|[RGBS](?:[2-9]|10|B|D|K|A))$")
PLAYER_RE = re.compile(r"^\((\d)\)(.*)$")
SCORES_RE = re.compile(r"^Ergebnis:\s*(-?\d+)\s*-\s*(-?\d+)\s*$")


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class LoggedTrade:
    giver: int
    receiver: int
    card: str


@dataclass(frozen=True)
class LoggedEvent:
    kind: str
    player: int | None = None
    cards: tuple[str, ...] = ()
    value: str | int | None = None


@dataclass
class LoggedRound:
    players: dict[int, str] = field(default_factory=dict)
    grand_hands: dict[int, tuple[str, ...]] = field(default_factory=dict)
    start_hands: dict[int, tuple[str, ...]] = field(default_factory=dict)
    trades: list[LoggedTrade] = field(default_factory=list)
    events: list[LoggedEvent] = field(default_factory=list)
    grand_tichu: set[int] = field(default_factory=set)
    tichu: set[int] = field(default_factory=set)
    bomb_holders: set[int] = field(default_factory=set)
    scores: tuple[int, int] | None = None


@dataclass
class LoggedGame:
    players: dict[int, str]
    rounds: list[LoggedRound]


@dataclass(frozen=True)
class ValidationIssue:
    round_index: int
    code: str
    detail: str


def parse_player(token):
    match = PLAYER_RE.match(token.strip())
    if not match:
        raise ParseError("invalid player token: {!r}".format(token))
    return int(match.group(1)), match.group(2)


def _validate_card_token(token):
    if not CARD_TOKEN_RE.match(token):
        raise ParseError("invalid card token: {!r}".format(token))
    return token


def _parse_hand(line):
    tokens = line.split()
    if len(tokens) < 2:
        raise ParseError("invalid hand line: {!r}".format(line))
    player, name = parse_player(tokens[0])
    return player, name, tuple(_validate_card_token(token) for token in tokens[1:])


def _parse_announcement(line):
    _, player_token = line.split(":", 1)
    player, _ = parse_player(player_token.strip())
    return player


def _parse_all_announced_players(line):
    players = [int(value) for value in re.findall(r"\((\d)\)", line)]
    if not players:
        raise ParseError("missing player in line: {!r}".format(line))
    return players


def _parse_trade(line, players_by_name):
    giver_token, body = line.split(" gibt:", 1)
    giver, _ = parse_player(giver_token)
    trades = []
    body = body.strip()
    if body.endswith("-"):
        body = body[:-1].rstrip()
    for part in body.split(" - "):
        part = part.strip()
        if not part:
            continue
        receiver_name, card = part.rsplit(":", 1)
        receiver_name = receiver_name.strip()
        if receiver_name not in players_by_name:
            raise ParseError("unknown trade receiver: {!r}".format(receiver_name))
        trades.append(
            LoggedTrade(
                giver=giver,
                receiver=players_by_name[receiver_name],
                card=_validate_card_token(card.strip()),
            )
        )
    if len(trades) != 3:
        raise ParseError("expected three trades: {!r}".format(line))
    return trades


def parse_game(text):
    """Parse one complete or partially downloaded game log."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    players = {}
    rounds = []
    current = None
    phase = None
    last_play_player = None
    index = 0

    while index < len(lines):
        line = lines[index]

        if line == GRAND_HEADER:
            if current is not None and current.scores is None:
                rounds.append(current)
            current = LoggedRound()
            phase = "grand"
            last_play_player = None
            for hand_line in lines[index + 1:index + 5]:
                player, name, cards = _parse_hand(hand_line)
                current.grand_hands[player] = cards
                current.players[player] = name
            if not players:
                players.update(current.players)
            index += 5
            continue

        if current is None:
            raise ParseError("content found before first round: {!r}".format(line))

        if line == START_HEADER:
            phase = "start"
            for hand_line in lines[index + 1:index + 5]:
                player, name, cards = _parse_hand(hand_line)
                current.start_hands[player] = cards
                if current.players.get(player) != name:
                    raise ParseError("player name mismatch: {!r}".format(hand_line))
            index += 5
            continue

        if line == "Schupfen:":
            phase = "trade"
            trade_lines = []
            next_index = index + 1
            while next_index < len(lines) and " gibt:" in lines[next_index]:
                trade_lines.append(lines[next_index])
                giver_token = lines[next_index].split(" gibt:", 1)[0]
                giver, name = parse_player(giver_token)
                current.players[giver] = name
                next_index += 1
            by_name = {name: player for player, name in current.players.items()}
            for trade_line in trade_lines:
                current.trades.extend(_parse_trade(trade_line, by_name))
            index = next_index
            continue
        elif line == PLAY_HEADER:
            phase = "play"
        elif line.startswith("Grosses Tichu:"):
            player = _parse_announcement(line)
            current.grand_tichu.add(player)
            current.events.append(LoggedEvent("grand_tichu", player=player))
        elif line.startswith("Tichu:"):
            player = _parse_announcement(line)
            current.tichu.add(player)
            current.events.append(LoggedEvent("tichu", player=player))
        elif line.startswith("BOMBE:"):
            for player in _parse_all_announced_players(line):
                current.bomb_holders.add(player)
                current.events.append(LoggedEvent("bomb_notice", player=player))
        elif line.startswith("Wunsch:"):
            value = _validate_wish(line.split(":", 1)[1].strip())
            current.events.append(
                LoggedEvent("wish", player=last_play_player, value=value)
            )
        elif line.startswith("Drache an:"):
            player = _parse_announcement(line)
            current.events.append(LoggedEvent("dragon_to", player=player))
        elif line.startswith("Ergebnis:"):
            match = SCORES_RE.match(line)
            if not match:
                raise ParseError("invalid score line: {!r}".format(line))
            current.scores = (int(match.group(1)), int(match.group(2)))
            rounds.append(current)
            current = None
            phase = None
        elif phase == "play" and line.endswith(" passt."):
            player_token = line.split(None, 1)[0]
            player, _ = parse_player(player_token)
            current.events.append(LoggedEvent("pass", player=player))
        elif phase == "play" and line.startswith("("):
            player_token, body = line.split(":", 1)
            player, _ = parse_player(player_token)
            body = body.strip()
            cards = tuple(_validate_card_token(token) for token in body.split())
            current.events.append(LoggedEvent("play", player=player, cards=cards))
            last_play_player = player
        else:
            raise ParseError("unrecognized line in {} phase: {!r}".format(phase, line))
        index += 1

    if current is not None:
        rounds.append(current)
    if set(players) != {0, 1, 2, 3}:
        raise ParseError("expected four players")
    return LoggedGame(players=players, rounds=rounds)


def _validate_wish(value):
    if value not in {"2", "3", "4", "5", "6", "7", "8", "9", "10", "B", "D", "K", "A"}:
        raise ParseError("invalid wish: {!r}".format(value))
    return value


def final_hands(round_):
    """Return hands after all twelve exchanges."""
    hands = {player: set(cards) for player, cards in round_.start_hands.items()}
    for trade in round_.trades:
        if trade.card not in hands.get(trade.giver, set()):
            raise ParseError(
                "player {} cannot trade missing card {}".format(trade.giver, trade.card)
            )
        hands[trade.giver].remove(trade.card)
        hands[trade.receiver].add(trade.card)
    return {player: frozenset(cards) for player, cards in hands.items()}


def validate_game(game):
    """Perform format-level checks without assuming hidden game-engine rules."""
    issues = []
    for round_index, round_ in enumerate(game.rounds):
        grand_cards = [card for cards in round_.grand_hands.values() for card in cards]
        start_cards = [card for cards in round_.start_hands.values() for card in cards]
        if len(round_.grand_hands) != 4 or any(
            len(cards) != 8 for cards in round_.grand_hands.values()
        ):
            issues.append(ValidationIssue(round_index, "grand_hands", "expected 4 x 8 cards"))
        if len(round_.start_hands) != 4 or any(
            len(cards) != 14 for cards in round_.start_hands.values()
        ):
            issues.append(ValidationIssue(round_index, "start_hands", "expected 4 x 14 cards"))
        if len(grand_cards) != len(set(grand_cards)):
            issues.append(ValidationIssue(round_index, "duplicate_grand_card", "duplicate card"))
        if len(start_cards) != 56 or len(set(start_cards)) != 56:
            issues.append(ValidationIssue(round_index, "invalid_deck", "expected 56 unique cards"))
        for player in range(4):
            if not set(round_.grand_hands.get(player, ())).issubset(
                round_.start_hands.get(player, ())
            ):
                issues.append(
                    ValidationIssue(round_index, "grand_not_subset", "player {}".format(player))
                )
        if len(round_.trades) != 12:
            issues.append(
                ValidationIssue(round_index, "trade_count", "got {}".format(len(round_.trades)))
            )
        else:
            pairs = {(trade.giver, trade.receiver) for trade in round_.trades}
            expected = {(giver, receiver) for giver in range(4) for receiver in range(4) if giver != receiver}
            if pairs != expected:
                issues.append(ValidationIssue(round_index, "trade_pairs", "missing or duplicate pair"))
            try:
                hands = final_hands(round_)
                if any(len(hand) != 14 for hand in hands.values()):
                    issues.append(ValidationIssue(round_index, "post_trade_size", "expected 14 cards"))
            except ParseError as error:
                issues.append(ValidationIssue(round_index, "invalid_trade", str(error)))
        if round_.scores is None:
            issues.append(ValidationIssue(round_index, "incomplete_round", "missing result"))
    return issues


def parse_file(path):
    return parse_game(Path(path).read_text(encoding="utf-8", errors="replace"))
