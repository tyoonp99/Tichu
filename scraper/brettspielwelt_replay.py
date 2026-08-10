"""Replay parsed Brettspielwelt rounds with the local Tichu rules engine."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gym_tichu.envs.internals.cards import Card, CardRank, Combination, Straight
from gym_tichu.envs.internals.actions import (
    GiveDragonAwayAction,
    PassAction,
    PlayCombination,
    Trick,
    WinTrickAction,
    WishAction,
)
from gym_tichu.envs.internals.tichu_state import (
    HandCards,
    History,
    TichuState,
    WonTricks,
)

from scraper.brettspielwelt_parser import final_hands, parse_file, validate_game


SPECIAL_CARDS = {
    "Ph": Card.PHOENIX,
    "Dr": Card.DRAGON,
    "Ma": Card.MAHJONG,
    "Hu": Card.DOG,
}
SUIT_NAMES = {
    "R": "HOUSE",  # red stars; the engine retains the historical HOUSE name
    "G": "JADE",
    "B": "PAGODA",
    "S": "SWORD",
}
RANK_NAMES = {
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
    "10": "TEN",
    "B": "J",
    "D": "Q",
    "K": "K",
    "A": "A",
}
WISH_RANKS = {
    token: getattr(CardRank, rank_name) for token, rank_name in RANK_NAMES.items()
}


@dataclass
class ReplayResult:
    status: str
    decisions: list[tuple[object, object, object]] = field(default_factory=list)
    event_index: int | None = None
    detail: str | None = None
    expected_scores: tuple[int, int] | None = None
    actual_scores: tuple[int, int] | None = None


def card_from_token(token):
    if token in SPECIAL_CARDS:
        return SPECIAL_CARDS[token]
    suit_name = SUIT_NAMES[token[0]]
    rank_name = RANK_NAMES[token[1:]]
    return getattr(Card, "{}_{}".format(rank_name, suit_name))


def make_initial_state(round_):
    hands = final_hands(round_)
    engine_hands = [frozenset(card_from_token(token) for token in hands[player]) for player in range(4)]
    mahjong_players = [player for player, hand in enumerate(engine_hands) if Card.MAHJONG in hand]
    if len(mahjong_players) != 1:
        raise ValueError("expected exactly one Mahjong holder")
    return TichuState(
        player_pos=mahjong_players[0],
        handcards=HandCards(*engine_hands),
        won_tricks=WonTricks(),
        trick_on_table=Trick(),
        wish=None,
        ranking=(),
        announced_tichu=frozenset(round_.tichu),
        announced_grand_tichu=frozenset(round_.grand_tichu),
        history=History(),
        allow_tichu=False,
        allow_wish=True,
    )


def _action_kind(actions):
    if not actions:
        return "none"
    if all(isinstance(action, WishAction) for action in actions):
        return "wish"
    if all(isinstance(action, GiveDragonAwayAction) for action in actions):
        return "dragon"
    if all(isinstance(action, WinTrickAction) for action in actions):
        return "win_trick"
    return "play"


def _prepare_for_event(state, event_kind):
    """Apply engine-only administrative actions omitted from text logs."""
    while not state.is_terminal():
        actions = state.possible_actions_list
        kind = _action_kind(actions)
        if kind == "win_trick":
            state = state.next_state(actions[0])
            continue
        if kind == "wish" and event_kind != "wish":
            no_wish = next(action for action in actions if action.wish is None)
            state = state.next_state(no_wish)
            continue
        break
    return state


def _matching_action(state, event):
    actions = state.possible_actions_list
    if event.kind == "pass":
        matches = [
            action
            for action in actions
            if isinstance(action, PassAction) and action.player_pos == event.player
        ]
    elif event.kind == "play":
        cards = frozenset(card_from_token(token) for token in event.cards)
        matches = [
            action
            for action in actions
            if isinstance(action, PlayCombination)
            and action.player_pos == event.player
            and frozenset(action.combination.cards) == cards
        ]
        if not matches:
            matches = _equivalent_physical_actions(state, event.player, cards)
    elif event.kind == "wish":
        rank = WISH_RANKS[event.value]
        matches = [
            action
            for action in actions
            if isinstance(action, WishAction)
            and action.player_pos == event.player
            and action.wish == rank
        ]
    elif event.kind == "dragon_to":
        matches = [
            action
            for action in actions
            if isinstance(action, GiveDragonAwayAction) and action.to == event.player
        ]
    else:
        return None, []
    return (matches[0] if matches else None), matches


def _rank_signature(cards):
    return tuple(sorted(card.rank.value for card in cards if card is not Card.PHOENIX))


def _equivalent_physical_actions(state, player, cards):
    """Recover legal suit variants omitted by the engine's action generator.

    For ordinary combinations suits do not determine strength, but the legacy
    generator emits one representative card per rank.  The actual physical
    cards still matter for future bombs, so replay constructs the exact logged
    combination after matching it to an otherwise equivalent legal action.
    """
    if not cards.issubset(state.handcards[player]):
        return []
    signature = _rank_signature(cards)
    recovered = []
    for candidate in state.possible_actions_list:
        if not isinstance(candidate, PlayCombination) or candidate.player_pos != player:
            continue
        if len(candidate.combination.cards) != len(cards):
            continue
        if _rank_signature(candidate.combination.cards) != signature:
            continue
        try:
            if isinstance(candidate.combination, Straight) and Card.PHOENIX in cards:
                combination = Straight(
                    cards, phoenix_as=candidate.combination.phoenix_as
                )
            else:
                combination = Combination.make(cards)
            if type(combination) is not type(candidate.combination):
                continue
            recovered.append(
                candidate.__class__(player_pos=player, combination=combination)
            )
        except (AssertionError, TypeError, ValueError):
            continue
    return recovered


def _apply_action(state, action):
    if action in state.possible_actions_set:
        return state.next_state(action)
    if isinstance(action, PlayCombination):
        # The action was recovered as a physical suit variant of a generated
        # legal combination.  The normal transition is still used; only the
        # legacy membership check is bypassed.
        return state._next_state_on_combination(action)
    raise ValueError("recovered non-play action is not legal")


def canonical_action(state, action):
    """Map an exact logged suit choice to the engine's legal representative."""
    if action in state.possible_actions_set:
        return action
    if not isinstance(action, PlayCombination):
        return None
    signature = _rank_signature(action.combination.cards)
    for candidate in state.possible_actions_list:
        if (
            isinstance(candidate, PlayCombination)
            and candidate.player_pos == action.player_pos
            and type(candidate.combination) is type(action.combination)
            and len(candidate.combination.cards) == len(action.combination.cards)
            and _rank_signature(candidate.combination.cards) == signature
            and candidate.combination.height == action.combination.height
        ):
            return candidate
    return None


def replay_round(round_):
    """Replay one round and report the first incompatibility, if any."""
    if round_.scores is None:
        return ReplayResult(status="incomplete_round")
    issues = validate_game(type("Game", (), {"rounds": [round_]})())
    if issues:
        return ReplayResult(status="invalid_structure", detail=issues[0].code)

    try:
        state = make_initial_state(round_)
    except Exception as error:
        return ReplayResult(status="setup_error", detail=str(error))

    decisions = []
    ignored = {"grand_tichu", "tichu", "bomb_notice"}
    for event_index, event in enumerate(round_.events):
        if event.kind in ignored:
            continue
        try:
            pending_kind = _action_kind(state.possible_actions_list)
            if (
                event.kind == "pass"
                and pending_kind in {"win_trick", "dragon"}
                and state.possible_actions_list[0].player_pos == event.player
            ):
                # BSW explicitly records the leading player's final pass.  In
                # this engine the trick has already finished at that point.
                continue
            state = _prepare_for_event(state, event.kind)
            if state.is_terminal():
                if event.kind in {"wish", "dragon_to"}:
                    # BSW may print the administrative result of the last play
                    # after a double win has already ended the round.
                    continue
                return ReplayResult(
                    status="event_after_terminal",
                    decisions=decisions,
                    event_index=event_index,
                    detail=event.kind,
                )
            action, matches = _matching_action(state, event)
            if action is None:
                return ReplayResult(
                    status="illegal_{}".format(event.kind),
                    decisions=decisions,
                    event_index=event_index,
                    detail="player={} cards={} expected={}".format(
                        event.player,
                        event.cards,
                        _action_kind(state.possible_actions_list),
                    ),
                )
            if event.kind in {"play", "pass"}:
                decisions.append((state, event, action))
            state = _apply_action(state, action)
        except Exception as error:
            return ReplayResult(
                status="engine_error",
                decisions=decisions,
                event_index=event_index,
                detail="{}: {}".format(type(error).__name__, error),
            )

    try:
        state = _prepare_for_event(state, "end")
        if not state.is_terminal():
            return ReplayResult(
                status="not_terminal",
                decisions=decisions,
                detail="ranking={} hand_sizes={}".format(
                    state.ranking, [len(hand) for hand in state.handcards]
                ),
            )
        points = state.count_points()
        actual = (points[0], points[1])
        status = "ok" if actual == round_.scores else "score_mismatch"
        return ReplayResult(
            status=status,
            decisions=decisions,
            expected_scores=round_.scores,
            actual_scores=actual,
        )
    except Exception as error:
        return ReplayResult(
            status="engine_error",
            decisions=decisions,
            detail="{}: {}".format(type(error).__name__, error),
        )


def scan_directory(input_dir, report_path=None):
    counts = Counter()
    games = rounds = decisions = 0
    output = None
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output = report_path.open("w", encoding="utf-8", newline="\n")
    try:
        for path in sorted(Path(input_dir).glob("*.tch")):
            game = parse_file(path)
            games += 1
            for round_index, round_ in enumerate(game.rounds):
                result = replay_round(round_)
                rounds += 1
                decisions += len(result.decisions)
                counts[result.status] += 1
                if output:
                    output.write(
                        json.dumps(
                            {
                                "game_id": int(path.stem),
                                "round": round_index,
                                "status": result.status,
                                "event_index": result.event_index,
                                "detail": result.detail,
                                "decisions": len(result.decisions),
                                "expected_scores": result.expected_scores,
                                "actual_scores": result.actual_scores,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    finally:
        if output:
            output.close()
    return {
        "games": games,
        "rounds": rounds,
        "decisions_before_failure": decisions,
        "counts": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay Brettspielwelt logs locally")
    parser.add_argument("--input-dir", default="datasets/raw/brettspielwelt")
    parser.add_argument(
        "--report", default="datasets/processed/brettspielwelt-replay-report.jsonl"
    )
    args = parser.parse_args()
    print(json.dumps(scan_directory(args.input_dir, args.report), sort_keys=True))


if __name__ == "__main__":
    main()
